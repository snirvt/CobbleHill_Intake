"""Diagnosis-category prompt text shared by every classifier that assigns a bucket."""

from classifier.models import DiagnosisCategory

# Bucket list shown in the prompt, one per line, derived from the enum.
CATEGORY_LIST = "\n".join(f"- {c.value}" for c in DiagnosisCategory)

# How to pick a bucket; shared so the main-diagnosis and per-diagnosis buckets agree.
CATEGORY_RULES = """\
- Pick the most specific category: a hip fracture or fall is "Fracture / Fall", not \
"Musculoskeletal / Spine"; a subdural or subarachnoid bleed is "Brain Bleed / Stroke", \
not "Neurological".
- Reflux/GERD (including in infants) is "Reflux / GERD", not "Feeding / Growth".
- "Infection / Sepsis" only when a systemic infection is the main problem; an infection \
of one organ goes to that organ (pneumonia → "Respiratory", UTI → "Kidney / Urinary").
- Postpartum depression is "Mental Health"; other postpartum or pregnancy conditions \
(e.g. postpartum hemorrhage, preeclampsia, postpartum care) are "Pregnancy / Postpartum".
- Dementia, Alzheimer's disease and cognitive decline in adults are "Mental Health"; \
developmental delay or intellectual disability in a child is "Developmental / Cognitive".
- Venous stasis, venous insufficiency, chronic leg edema and venous stasis ulcers are \
"Lower Extremity / Venous", not "Cardiovascular", "Skin" or "Wound".
- Any other wound or ulcer (pressure injury, diabetic foot ulcer, surgical or traumatic \
wound) is "Wound"; rashes, dermatitis and other skin conditions are "Skin".
- Use "Other" only when a diagnosis exists but fits no category."""
