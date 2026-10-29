"""Diagnosis-category prompt text shared by every classifier that assigns a bucket."""

from classifier.models import DiagnosisCategory

# Bucket list shown in the prompt, one per line, derived from the enum.
CATEGORY_LIST = "\n".join(f"- {c.value}" for c in DiagnosisCategory)

# How to pick a bucket; shared so the main-diagnosis and per-diagnosis buckets agree.
CATEGORY_RULES = """\
- Pick the most specific category: a hip fracture or fall is "Fracture / Fall", not \
"Musculoskeletal / Spine"; a subdural or subarachnoid bleed is "Brain Bleed / Stroke", \
not "Neurological".
- Reflux, heartburn or GERD (including in infants) is "Reflux / GERD", not "Feeding / Growth"; \
never use "Reflux / GERD" for uterine, pelvic or general abdominal pain.
- Feeding problems and growth concerns are "Feeding / Growth" even in a newborn or infant \
(e.g. feeding difficulty, poor feeding, poor weight gain, failure to thrive, slow weight \
gain); "Newborn / Well Child" is only for routine newborn care or well-child visits with \
no specific problem.
- "Infection / Sepsis" only when a systemic infection is the main problem; an infection \
of one organ goes to that organ (pneumonia → "Respiratory", UTI → "Kidney / Urinary").
- Postpartum depression is "Mental Health"; any other pregnancy, postpartum or \
gynecological condition or symptom (e.g. uterine or pelvic pain, menstrual problems, \
abnormal vaginal bleeding, fibroids, endometriosis, ovarian cyst, contractions, postpartum \
hemorrhage, preeclampsia, postpartum care) is "Women's Health", not "Reflux / GERD", \
"Gastrointestinal" or "Kidney / Urinary".
- Dementia, Alzheimer's disease and cognitive decline in adults are "Mental Health"; \
developmental delay or intellectual disability in a child is "Developmental / Cognitive".
- Venous stasis, venous insufficiency, chronic leg edema and venous stasis ulcers are \
"Lower Extremity / Venous", not "Cardiovascular", "Skin" or "Wound".
- Any other wound or ulcer (pressure injury, diabetic foot ulcer, surgical or traumatic \
wound) is "Wound"; rashes, dermatitis and other skin conditions are "Skin".
- Use "Other" only when a diagnosis exists but fits no category."""
