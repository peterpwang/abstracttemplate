""" Conditional text generation with the auto-regressive models of GPT-2 and PPLM
"""


import sys
import argparse
import logging
import readline

import numpy as np
import torch

import stanza

sys.path.insert(0, './pplm')
import generate_pplm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer)
}

#stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def rlinput(prompt, prefill=''):
   readline.set_startup_hook(lambda: readline.insert_text(prefill))
   try:
      return input(prompt)  # or raw_input in Python 2
   finally:
      readline.set_startup_hook()


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def create_upos(line, prompt_text):

    # Export documents into plain text and return in list
    doc = nlp(line)
    s = ""
    for sentence in doc.sentences:
        previous_ner = False
        for token in sentence.tokens:
            if token.ner != "O" and token.text not in prompt_text:
                #if not previous_ner:
                s += '<UNK> ' ##[[[' + token.text + ']]] '
                previous_ner = True
            else:
                s += token.text + ' '
                previous_ner = False
    return s


def remove_uncompleted_sentence(line):

    text = ""
    doc = nlp(line)
    if len(doc.sentences) == 1:
        text = doc.sentences[0].text
    else:
        for i in range(len(doc.sentences)-1):
            text = text + doc.sentences[i].text
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

    args = parser.parse_args()

    # Generate
    run_pplm(**vars(args))

"""
            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]
"""


def run_pplm(
        pretrained_model="gpt2-medium",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular'
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            print("discrim = {}, pretrained_model set "
            "to discriminator's = {}".format(discrim, pretrained_model))

    # Initialize the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") ##pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    prompt_text = cond_text if cond_text else ''

    while(True):
        # Accept initial prompt
        print()
        prompt_text = rlinput("Model prompt >>> ", prompt_text)

        # figure out conditioning text
        if uncond:
            tokenized_cond_text = tokenizer.encode(
                [tokenizer.bos_token],
                add_special_tokens=False
            )
        else:
            tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + prompt_text,
                add_special_tokens=False
            )

        # generate unperturbed and perturbed texts

        # full_text_generation returns:
        # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
        unpert_gen_tok_text, pert_gen_tok_texts, _, _ = generate_pplm.full_text_generation(
            model=model,
            tokenizer=tokenizer,
            context=tokenized_cond_text,
            device=device,
            num_samples=num_samples,
            bag_of_words=bag_of_words,
            discrim=discrim,
            class_label=class_label,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale
        )

        # untokenize unperturbed text
        #unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])
        #
        #print("=" * 80)
        #print("= Unperturbed generated text =")
        #print(unpert_gen_text)
        #print()

        generated_texts = []

        bow_word_ids = set()
        if bag_of_words and colorama:
            bow_indices = generate_pplm.get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
            for single_bow_list in bow_indices:
                # filtering all words in the list composed of more than 1 token
                filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
                # w[0] because we are sure w has only 1 item because previous fitler
                bow_word_ids.update(w[0] for w in filtered)

        # iterate through the perturbed texts
        generated_sequences = []
        for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
            try:
                # untokenize unperturbed text
                if colorama:
                    import colorama

                    pert_gen_text = ''
                    for word_id in pert_gen_tok_text.tolist()[0]:
                        if word_id in bow_word_ids:
                            pert_gen_text += '{}{}{}'.format(
                                colorama.Fore.RED,
                                tokenizer.decode([word_id]),
                                colorama.Style.RESET_ALL
                            )
                        else:
                            pert_gen_text += tokenizer.decode([word_id])
                else:
                    pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

                print("=== GENERATED TEXT {} ===".format(i + 1))
                generated_text = pert_gen_text.replace("<|endoftext|>","") 
                #generated_text = remove_uncompleted_sentence(generated_text)
                print("..." + generated_text)
                generated_text = create_upos(generated_text, prompt_text)
                generated_sequences.append(generated_text)
                #print()
            except:
                pass

            # keep the prefix, perturbed seq, original seq for each index
            generated_texts.append(
                (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
            )

        print("=" * 80)
        while(True):
            option_string = input("Select number:")
            if (option_string.isdigit() and int(option_string)>0 and int(option_string)<=len(generated_sequences)):
                prompt_text = generated_sequences[int(option_string)-1]
                break
    # End of while(True)


if __name__ == "__main__":
    main()

