"""Diagnosis-category prompt text shared by every classifier that assigns a bucket."""

from classifier.models import DiagnosisCategory
from classifier.output.csv_writer import format_age
from config.settings import settings

# Bucket list shown in the prompt, one per line, derived from the enum.
CATEGORY_LIST = "\n".join(f"- {c.value}" for c in DiagnosisCategory)

# How to pick a bucket; shared so the main-diagnosis and per-diagnosis buckets agree.
CATEGORY_RULES = """\
- Pick the most specific category: a hip fracture or fall is "Fracture / Fall", not \
"Musculoskeletal / Spine"; a subdural or subarachnoid bleed is "Brain Bleed / Stroke", \
not "Neurological".
- Reflux, heartburn or GERD (including in infants) is "Reflux / GERD / Vomiting", not \
"Feeding / Growth"; never use "Reflux / GERD / Vomiting" for uterine, pelvic or general \
abdominal pain.
- Vomiting, emesis or spit-up at any age (e.g. vomiting in infants, persistent vomiting) is \
"Reflux / GERD / Vomiting", not "Infection / Sepsis" or "Gastrointestinal".
- Growth and feeding problems are "Feeding / Growth" at any age: short stature (R62.52), \
growth delay or lack of expected growth (R62.5x), feeding difficulty or feeding disability \
(R63.3x), poor feeding, failure to thrive (R62.51); never "Gastrointestinal", \
"Blood / Anemia" or "Developmental / Cognitive".
- "Newborn / Well Child" is only for routine newborn care or a well-child visit of an infant \
or child with no specific problem; never for an adult, and never for a specific problem such \
as feeding or growth (→ "Feeding / Growth").
- A negative finding, ruled-out condition or normal status (e.g. no respiratory symptoms, \
no evidence of tuberculosis, incision healing well) is "Other", not the organ or infection \
category it mentions.
- Weight problems (underweight or low weight (R63.6), poor or slow weight gain, weight loss \
(R63.4), malnutrition) are "Feeding / Growth" when the patient is an infant or the age is \
unknown (the infant label is shown next to the patient age). Only when the patient is \
labelled not an infant are they "Obesity / Metabolic", unless the diagnosis itself names \
a feeding problem (e.g. feeding difficulty, feeding tube dependence) → "Feeding / Growth".
- "Obes / Metabolic" also covers obesity, overweight, diabetes, thyroid disease and \
high cholesterol.
- Fussiness, fussy baby, irritability, irritable mood, neonatal agitation, excessive crying \
and colic are "Fussiness / Irritability", not "Brain Bleed / Stroke", "Neurological", \
"Mental Health" or "Gastrointestinal".
- "Infection / Sepsis" only when a systemic infection is the main problem; an infection \
of one organ goes to that organ (pneumonia → "Respiratory", UTI → "Kidney / Urinary").
- Postpartum depression is "Mental Health"; any other pregnancy, postpartum or \
gynecological condition or symptom (e.g. uterine or pelvic pain, menstrual problems, \
abnormal vaginal bleeding, fibroids, endometriosis, ovarian cyst, contractions, postpartum \
hemorrhage, preeclampsia, postpartum care) is "Women's Health", not "Reflux / GERD / Vomiting", \
"Gastrointestinal" or "Kidney / Urinary".
- Dementia, Alzheimer's disease and cognitive decline in adults are "Mental Health"; \
developmental delay or intellectual disability in a child is "Developmental / Cognitive" \
(a feeding disability is "Feeding / Growth", not developmental).
- Venous stasis, venous insufficiency, chronic leg edema and venous stasis ulcers are \
"Lower Extremity / Venous", not "Cardiovascular", "Skin" or "Wound".
- Any other wound or ulcer (pressure injury, diabetic foot ulcer, surgical or traumatic \
wound) is "Wound"; rashes, dermatitis and other skin conditions are "Skin".
- Use "Other" only when a diagnosis exists but fits no category."""

ity
# Worked examples for the bucketing prompt (synthetic): "diagnosis, age → category".
CATEGORY_EXAMPLES = """\
- Short stature (R62.52), 5 years (not an infant) → Feeding / Growth
- Underweight (R63.6), 0 days (infant) → Feeding / Growth
- Underweight (R63.6), 40 years (not an infant) → Obesity / Metabolic
- Vomiting in infants (R11.10), 2 months (infant) → Reflux / GERD / Vomiting
- Neonatal agitation / fussy infant (R68.12), 1 month (infant) → Fussiness / Irritability
- Abdominal bloating (R14.0), 0 days (infant) → Gastrointestinal
- No respiratory symptoms at this time, 75 years (not an infant) → Other"""


def describe_age(age_years: float | None) -> str | None:
    """Age text for the bucketing prompt: "3 months (infant)" / "16 years (not an infant)".

    The infant cutoff is settings.infant_age_threshold_years; only the label reaches the
    model. None when the age is unknown.
    """
    if age_years is None or age_years < 0:
        return None
    label = "infant" if age_years < settings.infant_age_threshold_years else "not an infant"
    return f"{format_age(age_years)} ({label})"
