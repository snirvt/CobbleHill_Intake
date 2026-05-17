import re

from classifier.models import ExaminationData, ExtractedFields, MedicationData, PatientMetadata


class ProgressNoteExtractor:
    """Extracts structured fields from progress note text using regex.

    Handles OCR noise from scanned fax documents (e.g. 'Ace No.' for 'Acc No.').
    """

    # "Patient: Last, First" — name ends at 3+ spaces (provider info follows on same line)
    _NAME = re.compile(r"Patient:\s*([\w\s,.']+?)(?:\s{3,}|$)", re.MULTILINE)

    # DOB from labeled field: "DOB: MM/DD/YYYY"
    _DOB = re.compile(r"DOB:\s*([\d/]+)")

    # Inline header age+sex: "(8 mo F)" or "(2 yr M)"
    _AGE_SEX_INLINE = re.compile(
        r"\((\d+\s*(?:mo|yr|y|m)\w*)\s+(M|F)\)",
        re.IGNORECASE,
    )

    # Fallback labeled sex: "sex: Female"
    _SEX_LABEL = re.compile(r"\bsex:\s*(Male|Female)\b", re.IGNORECASE)

    # Account number: handles "Acc No. 399854", "Ace No. 399854", "Account Number: 399854"
    _ACCOUNT = re.compile(
        r"(?:Account\s*Number[:\s]+|Ac[ce]+\w*\s+No\.?\s+)(\w+)",
        re.IGNORECASE,
    )

    # Date of service: "DOS: MM/DD/YYYY"
    _DOS = re.compile(r"DOS:\s*([\d/]+)")

    # Phone: "Phone: 347-449-0280"
    _PHONE = re.compile(r"Phone:\s*([\d\-]+)")

    # Address: "Address, 380 Henry St..." or "Address: ..."
    _ADDRESS = re.compile(r"Address[,:]?\s*([^\n]+)")

    # Detects a new section header: short line, starts uppercase, ends with : or ,
    _SECTION_HEADER = re.compile(r"^[A-Z][\w\s/\-]*[:\,]\s*$")

    # Medication sub-section headers
    _MED_TAKING = re.compile(r"^Taking\s*$", re.IGNORECASE)
    _MED_NOT_TAKING = re.compile(r"^Not[\s\-]*[Tt]aking", re.IGNORECASE)

    # Fax/EMR artifact lines that are safe to strip globally — these never appear in clinical content.
    # "Document: Signed" is NOT here because OCR garbles it; handled per-section via extra_stops.
    _FAX_NOISE = re.compile(
        r"FROM:\s+Fax"           # fax transmission header
        r"|PAGE:\s*\d+\s+OF\s+\d+"  # fax page counter (e.g. PAGE: 007 OF 007)
        r"|eCW\s*\(",            # eClinicalWorks EMR system identifier
        re.IGNORECASE,
    )

    def extract(self, text: str) -> ExtractedFields:
        """Return all structured fields parsed from document text."""
        text = self._strip_fax_noise(text)
        age: str | None = None
        sex: str | None = None

        inline = self._AGE_SEX_INLINE.search(text)
        if inline:
            age = inline.group(1).strip()
            sex = inline.group(2).upper()
        else:
            sex_label = self._SEX_LABEL.search(text)
            if sex_label:
                sex = sex_label.group(1).capitalize()

        meta = PatientMetadata(
            patient_name=self._first(self._NAME, text),
            dob=self._first(self._DOB, text),
            age=age,
            sex=sex,
            account_number=self._first(self._ACCOUNT, text),
            dos=self._first(self._DOS, text),
            phone=self._first(self._PHONE, text),
            address=self._first(self._ADDRESS, text),
        )
        ped_bullets = self._extract_bullets(
            r"Pediat\w*\s+Exam", text, extra_stops=[r"General\s+E"]
        )
        gen_text = self._extract_text_block(
            r"General\s+E\w*", text, extra_stops=[r"Assessment:"]
        )
        examination = ExaminationData(pediatric=ped_bullets, general=gen_text)

        return ExtractedFields(
            meta=meta,
            # ROS header is OCR-merged with its first word ("ROS:Pediatric"), so stop on "ROS:"
            hpi=self._extract_text_block("HPI", text, extra_stops=[r"ROS:"]),
            examination=examination if (ped_bullets or gen_text) else None,
            complaints=self._extract_bullets("Chief Complaints", text),
            medical_history=self._extract_bullets("Medical History", text),
            surgical_history=self._extract_bullets("Surgical History", text),
            # OCR often garbles this header; match loosely on the stable middle tokens
            hospitalization=self._extract_bullets(
                r"Hosp\w+/Major Diagnostic\s+\w+", text
            ),
            assessment=self._extract_bullets("Assessment", text),
            # "E!xamination:" has OCR noise; not caught by _SECTION_HEADER — stop explicitly
            ros=self._extract_bullets("ROS", text, extra_stops=[r"xamination"]),
            medications=self._extract_medications(text),
            # OCR often drops or garbles the "D" in "Document" → ")ocument: Signed ..."
            # stop on the stable suffix to tolerate any leading-character corruption
            plan=self._extract_text_block("Plan", text, extra_stops=[r"ocument:\s+Signed"]),
            procedure_codes=self._extract_bullets("Procedure Codes", text),
            # "Follow Up: At 9 Months" and "care Plan:" (OCR lowercase) don't match
            # _SECTION_HEADER — stop on them explicitly
            preventive_medicine=self._extract_text_block(
                "Preventive Medicine",
                text,
                extra_stops=[r"Follow\s+Up:", r"[Cc]are\s+[Pp]lan:"],
            ),
        )

    def _strip_fax_noise(self, text: str) -> str:
        """Remove lines that are reliably fax/EMR artifacts, not clinical content."""
        return "\n".join(
            line for line in text.splitlines()
            if not self._FAX_NOISE.search(line)
        )

    def _extract_medications(self, text: str) -> MedicationData | None:
        """Parse the Medications section into taking/not-taking sub-lists.

        Handles bare bullets (• alone on a line, item text on next line) and
        stops at the 'Medication List reviewed' terminal sentence.
        """
        header_match = re.search(r"Medications.*?\n", text, re.IGNORECASE)
        if not header_match:
            return None

        taking: list[str] = []
        not_taking: list[str] = []
        current_list: list[str] | None = None
        current_parts: list[str] = []
        in_bullet = False

        for line in text[header_match.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "•" not in line and self._SECTION_HEADER.match(stripped):
                break
            if re.search(r"Medication\s+List", stripped, re.IGNORECASE):
                break
            if self._MED_TAKING.match(stripped):
                if in_bullet and current_parts and current_list is not None:
                    current_list.append(" ".join(current_parts).strip())
                current_parts = []
                in_bullet = False
                current_list = taking
                continue
            if self._MED_NOT_TAKING.match(stripped):
                if in_bullet and current_parts and current_list is not None:
                    current_list.append(" ".join(current_parts).strip())
                current_parts = []
                in_bullet = False
                current_list = not_taking
                continue
            if current_list is None:
                continue
            if "•" in line:
                if in_bullet and current_parts:
                    current_list.append(" ".join(current_parts).strip())
                after_bullet = re.sub(r"\s*•\s*", "", stripped, count=1).rstrip("\\").strip()
                current_parts = [after_bullet] if after_bullet else []
                in_bullet = True
            elif in_bullet:
                text_part = stripped.rstrip("\\").strip()
                if text_part:
                    current_parts.append(text_part)

        if in_bullet and current_parts and current_list is not None:
            current_list.append(" ".join(current_parts).strip())

        return MedicationData(
            taking=[t for t in taking if t],
            not_taking=[nt for nt in not_taking if nt],
        )

    def _extract_bullets(
        self,
        section_header_re: str,
        text: str,
        extra_stops: list[str] | None = None,
    ) -> list[str]:
        """Parse a bullet-point section, joining wrapped continuation lines.

        section_header_re is a raw regex. Header is matched with .*?\\n so OCR
        noise after the section name (e.g. '·' instead of ':') is tolerated.
        extra_stops: additional regex patterns that trigger end-of-section.
        """
        header_match = re.search(
            rf"(?:{section_header_re}).*?\n", text, re.IGNORECASE
        )
        if not header_match:
            return []

        items: list[str] = []
        current_parts: list[str] = []

        for line in text[header_match.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "•" not in line and self._SECTION_HEADER.match(stripped):
                break
            if extra_stops and any(
                re.search(p, stripped, re.IGNORECASE) for p in extra_stops
            ):
                break
            if "•" in line:
                if current_parts:
                    items.append(" ".join(current_parts))
                after_bullet = re.sub(r"\s*•\s*", "", stripped, count=1).rstrip("\\").strip()
                current_parts = [after_bullet] if after_bullet else []
            elif current_parts:
                current_parts.append(stripped.rstrip("\\").strip())

        if current_parts:
            items.append(" ".join(current_parts))

        return [item for item in items if item]

    def _extract_text_block(
        self, section_header_re: str, text: str, extra_stops: list[str] | None = None
    ) -> str | None:
        """Extract a free-form text section, joining all lines until the next section.

        Header matched with .*?\\n to tolerate OCR noise after section name.
        extra_stops: additional regex patterns that trigger end-of-section.
        """
        header_match = re.search(
            rf"(?:{section_header_re}).*?\n", text, re.IGNORECASE
        )
        if not header_match:
            return None

        lines: list[str] = []
        for line in text[header_match.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "•" not in line and self._SECTION_HEADER.match(stripped):
                break
            if extra_stops and any(
                re.search(p, stripped, re.IGNORECASE) for p in extra_stops
            ):
                break
            lines.append(stripped)

        return " ".join(lines) if lines else None

    @staticmethod
    def _first(pattern: re.Pattern[str], text: str) -> str | None:
        m = pattern.search(text)
        return m.group(1).strip() if m else None
