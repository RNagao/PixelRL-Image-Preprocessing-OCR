import pytesseract
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from rapidfuzz import fuzz
from jellyfish import hamming_distance, jaro_winkler_similarity, levenshtein_distance, damerau_levenshtein_distance

# Funcao que recebe um tensor e retorna as palavras lidas
def read_image_tensor_words(tensor: torch.Tensor):
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    text = pytesseract.image_to_string(image)
    return text


def read_image_array_words(img_array: np.ndarray):
    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
    text = pytesseract.image_to_string(pil_image)
    return text


def compare_strings_fuzzy(str1: str, str2: str, print_ratio:bool=False):
    ratio = fuzz.ratio(str1, str2)

    partial_ratio = fuzz.partial_ratio(str1, str2)

    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)

    token_set_ratio = fuzz.token_set_ratio(str1, str2)

    if print_ratio:
        print(f"Ratio: {ratio}")
        print(f"Partial ratio: {partial_ratio}")
        print(f"Token sort ratio: {token_sort_ratio}")
        print(f"Token set ratio: {token_set_ratio}")

    return ratio, partial_ratio, token_sort_ratio, token_set_ratio


def compare_strings_hamming(str1: str, str2: str, print_ratio=False):
    ratio = hamming_distance(str1, str2)

    if print_ratio:
        print(f"Distancia de Hamming: {ratio}")

    return ratio


def compare_strings_jaro_winkler(str1: str, str2: str, print_ratio: bool=False):
    ratio = jaro_winkler_similarity(str1, str2)

    if print_ratio:
        print(f"Distancia de Jaro Winkler: {ratio}")

    return ratio


def compare_strings_levenshtein(str1: str, str2: str, print_ratio:bool=False):
    ratio = levenshtein_distance(str1, str2)

    if print_ratio:
        print(f"Distancia de Levenshtein: {ratio}")

    return ratio


def compare_strings_damerau_levenshtein(str1: str, str2: str, print_ratio:bool=False):
    ratio = damerau_levenshtein_distance(str1, str2)

    if print_ratio:
        print(f"Distancia de Damerau Levenshtein: {ratio}")

    return ratio