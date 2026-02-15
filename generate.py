import torch
from utils import vectorize_string

def generate_text(model, start_string, char2idx, idx2char,
                  device, generation_length=500, temperature=0.8):

    model.eval()

    input_idx = vectorize_string(start_string, char2idx)
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)

    state = model.init_hidden(1, device)

    text_generated = []

    with torch.no_grad():
        for _ in range(generation_length):

            predictions, state = model(input_idx, state, return_state=True)
            predictions = predictions[:, -1, :]

            probs = torch.softmax(predictions / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1)

            text_generated.append(idx2char[next_idx.item()])
            input_idx = next_idx

    return start_string + "".join(text_generated)
