# common nlp tasks 

# text summazrization
# text/document classification
# sentiment analysis
# info retrieval
# search engines
# chatbots
# parts of speech tagging 
# question answering 
# lang detection
# machine translation
# conversational agents
# knowledge graphs
# transliteration
# topic modelling - abstract extraction of topics from  large text using lda algorithm
# text generation typos auto complete
# spell checking grammatical errors correction
# text parsing 
# speech-to-text



# approcaches to nlp 

# - heuristic methods   [ regular expressions , nltk wordnet lexical dict,  ]
# - ml methods          [ naive bayes, logistic regression, svm, lda, markov-chain models]  they dont care about sequential info retention and prevention , 
# - deep learning methods [ automatic features generation , unlike ml where we need to make on our own. ]
                        # [ rnn ] not been able to seq to seq  context retention
                        # [ lstm ] solved problem it can retain context of long sequences , sentences 
                        # [ gru/cnn ]  used for text generation , text classification 
                        # [ transformers ]  solved problem of context retention using self attention and muilti head attention 
                        # [ auto encoders ]
                        
                        
# challenges in nlp 

# AMBIGUITY  -  i saw the boy on the beach with my binoculars
#            -  i have never tasted a cake quite like that before 

# contextual words - i ran to the store because we ran out of milk 

# colloquialisms and slang - piece of cake , pulling your leg 
# synonyms
# irony,sarcasm, tonal difference  - that's just what i nedded today! 
# spelling errors 
# creativity     - poetry, dialogues, scripts
# diversity     - languages ,slang, etc.
# comdey jokes & figure of speech  types pun, hypreble 


# gpt 2 tokenizer 
# non eng lang work worse on chat gpt due to training data is much larger for eng then other lang same with tokenizer
# we're going to have a lot more longer tokens for eng 
# for eg. a sent in eng will have 10 tokens but its translation will have much more tokens than it 
# so we re using a lot more tokens for same thing it bloats up the seq len of all documents and goes out of the context window of attention of transformer and loses the context in the max context length of transformer  
# so all non text is stretched out of transformer perspective 
# same as with python that is why gpt 2 is not very good with python that is wastefull taking out too many tokens space  
# good way to yokenize is to make all indentations as one token 
# it squashes and makes i/p dense so that more words fit in context window and better next word prediction


# the no of tokens in cl100k_base (gpt 4 tokenizer) have roughly half the no of tokens for the excat same string. 
# thats coz no of tokens in gpt 4 are almost double the size of tokens in gpt 2


# increasing no of tokens strictly not good coz as u increase tokens number embedding table gots larger 
# and also at output we try to predict next word theres softmax there  which grows as well

# so there some kind of sweet spot where you have just right no of tokens in vocab everything is perfectly dense and still fairly efficient 
# gpt 4 tokenizer the handling of white space for python has improved a lot 
# (they group alot more whitespace into a single character ) it densifies and context length will have more words before to predict next word.


'''
tokenize strings into some integers into some fixed vocab
we will use these integers to see in lookup table of vectors and feed it into transformers

strings are immutable sequences of unified code points (150000 charac, emojis of all lang represented as numbers utf-8 encoding etc.)

ord("thy") 
[ ord(x) for x in "ewcwtw ercw werr !!" ] 
we can't just use them as it keeps on changing 
unicode defines 3 types of encodings ytf-8 utf-16 utf-32 

each code point is represented as few bytes 8,16, 32
list("hi my name is ctg korean guy ").encode("utf-8")  diff bytes representing string 
utf -8 are byte strings means vocab size of 256 charc which is very small all of our text will be stretched out very very long sequences 
so embedding tables will be small and prediction at top final layer will be small 
long seq  but finite context length (for computational reasons) in attention in transformers 
not allow us to attend sufficiently long text before next predicted token 
so we dont use raw bytes of utf-8 we want larger vocab size which can be used as hyperparam to tune 


BYTE PAIR ENCODING 

- allows to compress these byte sequences to var amount 
- aaabdaaabac vocab - 4; seq len 11 ;
- the byte pair "aa" occurs most freq so we replace that byte pair with single new token that we append to our vocab 

- ZabdZabac  {z=aa}  vocab size : 5 , seq len = 9 tokens 
- we find again a seq eg. 

ZYdZYac  {y=ab}  vocab size : 6 , seq len :7 tokens 

final round 
XdXac {X = ZY, Y=ab, Z=aa}

vocab size :7 , seq len : 5 tokens

we can iteratively compress our seq as we mint new tokens and append to our vocab
the data cannot be furthere compressed by bytepair encoding coz there are no more pair of bytes occuring more than once 

256 vocab size ==> bpe ==> vocabsize increases , token len decreases 

'''