import os
from unittest import TestCase
import tempfile
import shutil
import numpy as np
import numpy.testing as npt
from tk_nn_classifier.word_embedding.word_vector import WordVector, maxabs

class TestWordVector(TestCase):
    """Test the context"""
    @classmethod
    def setUpClass(self):
        embedding_content = '''5 3
FOO 0.1 1.0 -0.3
BAR 1.0 -0.6 1.0
ZOO -0.5 0.5 0.1
NEW 0.2 -1.0 0.7
OLD -1.0 0.9 -1.0
'''
        self.test_dir = tempfile.mkdtemp()
        self.txt_embedding_file = os.path.join(self.test_dir,
                                                'test_embedding.txt')

        with open(self.txt_embedding_file, 'w') as embedding_fh:
            embedding_fh.write(embedding_content)
        self.embedding_obj = WordVector(self.txt_embedding_file)
        self.expected_vector = [[ 0.00, 0.00, 0.00],
                                [ 0.00, 0.00, 0.00],
                                [ 0.09, 0.95, -0.28],
                                [ 0.65, -0.39, 0.65],
                                [-0.70, 0.70, 0.14],
                                [ 0.16, -0.80, 0.56],
                                [-0.59, 0.53, -0.59]]
        self.sample_embedding_bin_file = 'tests/resource/sample_embedding.bin'

    @classmethod
    def tearDownClass(self):
        '''clean up the temp dir after test'''
        shutil.rmtree(self.test_dir)

    def test_embedding_header_reading(self):
        txt_vocab_size, txt_vector_size = WordVector.read_embeddings_header(
            self.txt_embedding_file)
        self.assertEqual(txt_vocab_size, 5)
        self.assertEqual(txt_vector_size, 3)

        bin_vocab_size, bin_vector_size = WordVector.read_embeddings_header(
            self.sample_embedding_bin_file, 'binary')
        self.assertEqual(bin_vocab_size, 8852)
        self.assertEqual(bin_vector_size, 150)

    def test_embedding_reading_bin(self):
        embedding = WordVector(self.sample_embedding_bin_file)
        self.assertEqual([8854,150], list(embedding.vectors.shape))

    def test_embedding_reading_txt(self):
        self.assertEqual([7, 3], list(self.embedding_obj.vectors.shape))
        self.assertEqual(['xxPADxx', 'xxUNKxx', 'FOO', 'BAR', 'ZOO', 'NEW', 'OLD'],
                         self.embedding_obj.vocab.tolist() )
        #print(self.embedding_obj.vectors)
        npt.assert_almost_equal(self.embedding_obj.vectors,
                                self.expected_vector,
                                decimal=2)

    def test_embedding_properties(self):
        self.assertEqual(self.embedding_obj.vocab_size, 7)
        self.assertEqual(self.embedding_obj.vector_size, 3)
        npt.assert_equal(self.embedding_obj.unk_vector, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(self.embedding_obj.vocab_to_index,
                         {'xxPADxx': 0, 'xxUNKxx': 1, 'FOO': 2,
                          'BAR': 3, 'ZOO': 4, 'NEW': 5, 'OLD': 6},
                         'vocab to index mapping'
                         )

    def test_embedding_get_word(self):
        self.assertEqual(self.embedding_obj.get_index('NEW'), 5)
        self.assertEqual(self.embedding_obj.get_index('FOO'), 2)
        self.assertEqual(self.embedding_obj.get_word(2), 'FOO')
        self.assertEqual(self.embedding_obj.get_word(5), 'NEW')
        self.assertIn('FOO', self.embedding_obj)

    def test_get_vectors(self):
        words = ['FOO', 'AA', 'NEW', 'OLD']
        npt.assert_almost_equal(self.embedding_obj.get_vectors(words),
                                np.array(self.expected_vector)[[2,0,5,6], :],
                                decimal=2)

    def test_save_sublist(self):
        sub_list_file = os.path.join(self.test_dir, 'sub_list.bin')
        words = ['FOO', 'AA', 'NEW', 'OLD']
        self.embedding_obj.save_sublist(words, sub_list_file)
        sub_embedding = WordVector(sub_list_file)
        self.assertEqual(sub_embedding.vocab_size, 5)
        self.assertEqual(self.embedding_obj.vector_size, 3)
        self.assertEqual(['xxPADxx', 'xxUNKxx', 'FOO', 'NEW', 'OLD'],
                         sub_embedding.vocab.tolist() )

        npt.assert_almost_equal(sub_embedding.get_vectors(words),
                                np.array(self.expected_vector)[[2,0,5,6], :],
                                decimal=2)
        self.assertEqual(sub_embedding.get_word(2), 'FOO')
        self.assertEqual(sub_embedding.get_index('NEW'), 3)


    def test_maxabs(self):
        words = ['FOO', 'AA', 'NEW', 'OLD']
        maxabs_vec = maxabs(self.embedding_obj.get_vectors(words))
        npt.assert_almost_equal(maxabs_vec, [-0.59, 0.95, -0.59], decimal=2)
