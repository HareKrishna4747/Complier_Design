import nltk
from nltk import Tree
import matplotlib.pyplot as plt
import streamlit as st
import io
import networkx as nx

# Ensure you have the necessary NLTK resources downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import nltk
from nltk import Tree
import matplotlib.pyplot as plt
import streamlit as st
import io
import networkx as nx
import os

# Function to download NLTK data
def download_nltk_data():
    try:
        # Ensure the required NLTK resources are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Call the download function at the start
download_nltk_data()


def build_parse_tree(sentence):
    """Builds a parse tree from the given sentence using NLTK's parsing functions."""
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Tag the tokens with part of speech
    pos_tags = nltk.pos_tag(tokens)
    # Parse the POS tags using NLTK's named entity chunker
    parse_tree = nltk.ne_chunk(pos_tags)
    return parse_tree,tokens,pos_tags

def plot_tree(tree):
    """Plots the parse tree using Matplotlib and returns the figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Turn off the axis
    plot_tree_recursive(tree, ax)
    return fig

def plot_tree_recursive(tree, ax, x=0, y=0, level=1, x_offset=0.4):
    """Recursively plot the tree using Matplotlib"""
    node_name = tree.label() if isinstance(tree, Tree) else tree
    ax.text(x, y, node_name, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

    # If it's a Tree object, we recursively plot its children
    if isinstance(tree, Tree):
        num_children = len(tree)
        new_x_start = x - x_offset * (num_children - 1) / 2  # Start plotting the children
        for i, child in enumerate(tree):
            child_x = new_x_start + i * x_offset
            child_y = y - 0.1 * level
            # Draw a line from the current node to the child
            ax.plot([x, child_x], [y - 0.02, child_y + 0.02], 'k-')
            plot_tree_recursive(child, ax, child_x, child_y, level + 1, x_offset / num_children)
def plot_tokenization(tokens):
    """Plot the tokenization process using NetworkX."""
    G = nx.DiGraph()  # Create a directed graph

    # Add the main node for Tokenization
    G.add_node("Tokenization")

    # Add tokens as separate nodes and connect them to the "Tokenization" node
    for token in tokens:
        G.add_node(token)  # Add the token as a node
        G.add_edge("Tokenization", token)  # Connect the token to the "Tokenization" node

    # Create a layout and draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 6))

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgreen', font_size=10, font_color='black', font_weight='bold', arrows=True)
    
    plt.title('Tokenization Process', fontsize=16)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the plot to avoid display issues
    return buf

def plot_pos_tagging(pos_tags):
    """Plot the POS tagging process using NetworkX."""
    G = nx.DiGraph()  # Create a directed graph

    # Add the main node for POS tagging
    G.add_node("POS Tagging")

    # Add POS tags as separate nodes and connect them to the "POS Tagging" node
    for word, tag in pos_tags:
        tagged_word = f'{word}/{tag}'  # Create the tagged word string
        G.add_node(tagged_word)  # Add the tagged word as a node
        G.add_edge("POS Tagging", tagged_word)  # Connect the tagged word to the "POS Tagging" node

    # Create a layout and draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 6))

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
    
    plt.title('POS Tagging Process', fontsize=16)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the plot to avoid display issues
    return buf

def main():
    st.title("Parse Tree Visualizer")

    # Get user input
    sentence = st.text_input("Enter a sentence:")
    st.write("Created by Pragya and Jayaditya")

    if st.button("Generate Parse Tree"):
        if sentence:
            # Build the parse tree
            parse_tree,tokens,pos_tags = build_parse_tree(sentence)
            
            # Plot the tree
            fig = plot_tree(parse_tree)
            
            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            tokenization_buf = plot_tokenization(tokens)
            st.image(tokenization_buf)

            # Plot POS tagging
            pos_tagging_buf = plot_pos_tagging(pos_tags)
            st.image(pos_tagging_buf)
            
            # Display the plot in Streamlit
            st.image(buf)
            plt.close(fig)  # Close the figure to avoid display issues in Streamlit

        else:
            st.warning("Please enter a sentence.")
            

if __name__ == "__main__":
    main()
