class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    

    
    
    """def get_words_with_prefix(self, prefix):
        def dfs(node, path, results):
            if node.is_end_of_word:
                results.append(path)
            for char, next_node in node.children.items():
                dfs(next_node, path + char, results)

        results = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return results
            node = node.children[char]
        dfs(node, prefix, results)
        return results"""





































































































    def get_words_with_prefix(self, prefix):
        def dfs(node, path, results):
            if node.is_end_of_word:
                results.append(path)
            for char, next_node in node.children.items():
                dfs(next_node, path + char, results)

        results = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return results
            node = node.children[char]
        dfs(node, prefix, results)
        return results
