from Evaluate_Disc_BertScore import compute_E_disc, rescaled_BERTScore
def energy(x, seed, model, tokenizer, device, ALPHA_DISC=0.5, ALPHA_BERT=0.5):
    E_disc, _ = compute_E_disc(x, model, tokenizer, device)
    E_bert = -rescaled_BERTScore([x], [seed])[0]
    return ALPHA_DISC * E_disc + ALPHA_BERT * E_bert


def compute_inidvidual_Jscore(output, seed, model, tokenizer, device, formality_classifier):
    _, probs = compute_E_disc(output, model, tokenizer, device)
    acc = 1 if probs[1] > probs[0] else 0

    sim = rescaled_BERTScore([output], [seed])[0]

    formality_result = formality_classifier(output)[0]
    fl = 1 if formality_result['label'] == 'LABEL_1' else 0

    final_score = acc * sim * fl
    return final_score


def get_lowest_energy_sample(seed_chains):
    all_samples = []
    for chain in seed_chains:
        all_samples.extend(chain)  
            
    # Sort by energy (second element of tuple) and get the lowest
    if all_samples:
        best_sample = min(all_samples, key=lambda x: x[1])
        return best_sample[0]  # Return the sentence only
    return None


