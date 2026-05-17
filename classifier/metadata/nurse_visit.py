import re

from classifier.models import NurseVisitFields


class NurseVisitMetadataExtractor:
    """Extracts structured fields from nurse visit SOAP note text.

    Handles the tabular layout of Practice Fusion home-visit exports.
    """

    _DOS = re.compile(r"Date of service:\s*([\d/]+)", re.IGNORECASE)
    _PATIENT_HEADER = re.compile(r"Patient:\s*([\w\s,.']+?)\s+DOB:", re.IGNORECASE)
    _DOB = re.compile(r"\bDOB[:\s]+(\d{2}/\d{2}/\d{4})")
    _AGE = re.compile(r"^AGE\s+(\d+\s*\w+)", re.MULTILINE)
    _SEX = re.compile(r"^SEX\s+(Male|Female)", re.MULTILINE | re.IGNORECASE)
    _PRN = re.compile(r"\bPRN[:\s]+(\w+)")
    _SEEN_BY = re.compile(r"SEEN BY\s+(.+?)(?:\s{3,}|$)", re.MULTILINE)
    _ADDR1 = re.compile(r"^\s*ADDRESS LINE 1\s+(.+?)(?:\s{3,}|$)", re.MULTILINE)
    _CITY = re.compile(r"^\s*CITY\s+(\w[\w\s]*?)(?:\s{3,}|$)", re.MULTILINE)
    _STATE = re.compile(r"^\s*STATE\s+([A-Z]{2})\b", re.MULTILINE)
    _ZIP = re.compile(r"^\s*ZIP CODE\s+(\d+)", re.MULTILINE)
    _PHONE = re.compile(r"MOBILE PHONE\s+([\d()\-\s]+?)(?:\s{3,}|\n|$)", re.MULTILINE)

    # ICD code line: "(R63.30) Feeding difficulties..." or in assessment "Description – R63.3"
    _ICD_LINE = re.compile(r"^\s*\([A-Z]\d[\d.]+\)\s+.+", re.MULTILINE)
    _ASSESS_LINE = re.compile(r"^\s*\S.+?[–\-]\s*[A-Z]\d[\d.]+\s*$", re.MULTILINE)

    # Top-level section headers — used to detect end of a section being extracted.
    # Only real top-level sections here; sub-headers (PHYSICAL EXAM, GROWTH PARAMETERS)
    # are intentionally excluded so they don't prematurely terminate block extraction.
    _SECTION_HEADERS = re.compile(
        r"^\s*(?:Chief complaint|Diagnoses|Drug Allergies|Current Medications"
        r"|Past medical history|Health concern note|Active health concerns"
        r"|Inactive health concerns|Patient identifying details[\w\s]*"
        r"|Subjective|Objective|Assessment|Plan"
        r"|Orders|Screenings|Observations|Care plan)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def extract(self, text: str) -> NurseVisitFields:
        """Return all structured fields parsed from nurse visit text."""
        address = self._extract_address(text)

        phone_m = self._PHONE.search(text)
        phone = phone_m.group(1).strip() if phone_m else None

        return NurseVisitFields(
            patient_name=self._first(self._PATIENT_HEADER, text),
            dob=self._first(self._DOB, text),
            age=self._first(self._AGE, text),
            sex=self._first(self._SEX, text),
            prn=self._first(self._PRN, text),
            dos=self._first(self._DOS, text),
            address=address,
            phone=phone,
            seen_by=self._first(self._SEEN_BY, text),
            chief_complaint=self._extract_block("Chief complaint", text),
            diagnoses=self._extract_icd_list("Diagnoses", text),
            medications_active=self._extract_medications(text),
            subjective=self._extract_block("Subjective", text),
            objective=self._extract_block("Objective", text),
            assessment=self._extract_assessment(text),
            plan=self._extract_block("Plan", text),
            care_plan=self._extract_block("Care plan", text),
        )

    def _extract_address(self, text: str) -> str | None:
        addr1 = self._first(self._ADDR1, text)
        if not addr1:
            return None
        city = self._first(self._CITY, text)
        state = self._first(self._STATE, text)
        zip_code = self._first(self._ZIP, text)
        parts = [addr1]
        if city:
            parts.append(city)
        if state:
            parts.append(state)
        if zip_code:
            parts.append(zip_code)
        return ", ".join(parts)

    def _extract_block(self, header: str, text: str) -> str | None:
        """Extract free-form text under a section header until the next header."""
        m = re.search(rf"^\s*{re.escape(header)}\s*$", text, re.MULTILINE | re.IGNORECASE)
        if not m:
            return None
        body = text[m.end():]
        lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._SECTION_HEADERS.match(stripped):
                break
            # Skip URL artifacts from PDF export
            if stripped.startswith("http"):
                continue
            lines.append(stripped)
        return " ".join(lines) if lines else None

    def _extract_icd_list(self, header: str, text: str) -> list[str]:
        """Extract ICD-coded diagnosis lines from under a section header."""
        m = re.search(rf"^\s*{re.escape(header)}\s*$", text, re.MULTILINE | re.IGNORECASE)
        if not m:
            return []
        body = text[m.end():]
        items: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._SECTION_HEADERS.match(stripped):
                break
            if self._ICD_LINE.match(line):
                items.append(stripped)
        return items

    def _extract_assessment(self, text: str) -> list[str]:
        """Extract assessment lines (Description – ICD code format)."""
        m = re.search(r"^\s*Assessment\s*$", text, re.MULTILINE | re.IGNORECASE)
        if not m:
            return []
        body = text[m.end():]
        items: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._SECTION_HEADERS.match(stripped):
                break
            if stripped.startswith("http"):
                continue
            # Include lines with an ICD-like code at end
            if re.search(r"[–\-]\s*[A-Z]\d[\d.]+\s*$", stripped):
                items.append(stripped)
        return items

    def _extract_medications(self, text: str) -> list[str]:
        """Extract active medications listed under Current Medications section."""
        m = re.search(r"^Current Medications.*$", text, re.MULTILINE | re.IGNORECASE)
        if not m:
            return []
        body = text[m.end():]
        items: list[str] = []
        in_active = False
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^Active\s*$", stripped, re.IGNORECASE):
                in_active = True
                continue
            if re.match(r"^Historical\s*$", stripped, re.IGNORECASE):
                break
            if self._SECTION_HEADERS.match(stripped):
                break
            if in_active and stripped and not stripped.startswith("SIG") and not stripped.startswith("Was "):
                if "No active medications" not in stripped:
                    items.append(stripped)
        return items

    @staticmethod
    def _first(pattern: re.Pattern[str], text: str) -> str | None:
        m = pattern.search(text)
        return m.group(1).strip() if m else None
