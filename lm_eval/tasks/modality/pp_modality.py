def doc_to_text(doc) -> str:
    # Construct the options string
    option_choices = doc["options"]
    answers = "".join(f"{k}. {v}\n" for k, v in option_choices.items())
    # Create the prompt with the report and the options
    return f"{doc['report']}\nQuestion: What imaging modality was used? Note: 'MG' stands for Mammogram, 'US' for Ultrasound, 'MRI' for Magnetic Resonance Imaging, and 'BIO' for Biopsy. Hyphen (-) indicates multiple modalities.\n{answers}Answer:"

def doc_to_target(doc) -> str:
    # Return the correct answer's index/label (e.g., 'A')
    return doc["answer_idx"]
