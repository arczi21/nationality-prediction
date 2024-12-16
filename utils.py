import unicodedata


def remove_diacritics(text):
    normalized_text = unicodedata.normalize('NFD', text)
    stripped_text = ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')
    return stripped_text


class LetterEncoder:
    def __init__(self, alphabet_dictionary):
        self.alphabet_dictionary = alphabet_dictionary

    def encode_letter(self, letter):
        if letter in self.alphabet_dictionary:
            return self.alphabet_dictionary[letter]
        else:
            return self.alphabet_dictionary['other']


