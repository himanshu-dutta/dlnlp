from inference import prediction
from data import NounChunkDataset
from perceptron import RecurrentPerceptron

import streamlit as st
from argparse import Namespace

# Mapping of PoS tags
pos_mapping = {"NN": 1, "DT": 2, "JJ": 3, "OT": 4}
args = Namespace(num_inputs=9)


def main():
    st.title("Recurrent Perceptron for Noun Chunk Tagging")
    st.image("./artifacts/rnn.png")
    load_path = st.text_input(
        "Enter the path to load the model:", "artifacts/best_model.pkl"
    )
    st.success("Model loaded successfully!")

    model = RecurrentPerceptron(9)
    model.load_weights(load_path)
    print("Loaded model weights: ", model.dump_weights())

    input_sentence = st.text_input("Enter input sentence")
    if input_sentence:
        input_tokens = input_sentence.split()
        st.write(
            "<h2>Mark PoS Tags</h2>",
            unsafe_allow_html=True,
        )
        input_pos_tags = list()
        for idx, token in enumerate(input_tokens):
            col1, col2 = st.columns(2)
            col1.write(token)
            pos_option = col2.selectbox(
                f"PoS Tag: {token}",
                options=["NN", "DT", "JJ", "OT"],
                key=idx,
            )
            input_pos_tags.append(pos_option)
        pos_tagged_tokens = [f"{t}_{p}" for t, p in zip(input_tokens, input_pos_tags)]
        input_pos_tags = [pos_mapping[nm] for nm in input_pos_tags]

        if st.button("Submit"):
            st.write(
                "<h2>Token + PoS Tags</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(pos_tagged_tokens)
            if len(input_tokens) == len(input_pos_tags):
                try:
                    outputs = prediction([input_tokens], [input_pos_tags], args, model)
                    noun_chunk_tags = outputs[0].reshape((-1,)).tolist()
                    st.write(
                        "<h2>Noun Chunk Tags</h2>",
                        unsafe_allow_html=True,
                    )
                    noun_chunk_tagged_tokens = [
                        f"{t}_{n}" for t, n in zip(pos_tagged_tokens, noun_chunk_tags)
                    ]
                    st.markdown(noun_chunk_tagged_tokens)
                    color_sequence = NounChunkDataset.mark_nouns(
                        input_pos_tags,
                        NounChunkDataset.mark_noun_chunks(noun_chunk_tags),
                    )
                    output_token_str = ""
                    for token, noun_chunk_tag in zip(input_tokens, color_sequence):
                        if noun_chunk_tag == 1:
                            output_token_str += f'<span style="background-color:#891652">{token} </span>'
                        else:
                            output_token_str += f'<span style="background-color:#7755ff"><u>{token} </u></span>'
                    st.write(
                        "<h2>Tokens with Noun Chunk Tag Coloring:</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        '<h4 style="border: 2px solid #FB6D48; border-radius: 25px; text-align: center;">'
                        + output_token_str
                        + "</h4>",
                        unsafe_allow_html=True,
                    )

                    st.write(
                        '</br><span style="background-color:#7755ff">NOUN-CHUNK </span></br> <span style="background-color:#891652">N/A</span>',
                        unsafe_allow_html=True,
                    )
                except ValueError:
                    st.error(
                        "Invalid input. Please enter a valid binary string of 10 bits."
                    )
            else:
                st.error("Number of tokens and PoS tags do not match.")


if __name__ == "__main__":
    main()
