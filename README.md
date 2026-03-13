# Computational Linguistics

## Week 01

### Text

- How to get the text?
  - Text File
  - PDF File
  - Internet
  - API Call
  - RSS Feed 
- Data Clean
  - Depend on what do you want to study.
- Regular Expression (RegEx)
  - Language used to Specify and Match Patterns of Characters.
- Unicode
  - String Encryption 
  - Each character has its own code.
- Metacharacters
- Grouping
  - ภาษาไทยใช้ Unicode จาก ก ถึง การันต์ ไม่ก็ถึงตัวเลขไทย 
- Quantifier
- Three regexes 
  ```py
  import re
  re.sub(<Pattern Match>, <Replace with>, ...)
  ```
- Variation of Languages (Sociolinguistic Factors)
- Complexity of Language 
  - Language has structure
    - Discourse: List of Sentences
    - Syntax: Sentence and Phrase Structure
    - Morphology: Structure of a word.
  - Linguistics Structure
    - Phonetics 
    - Phonology
    - Morphology
    - Syntax 
    - Semantics
    - Pragmatics
- Common Linguistic Analysis
  - Word Segmentation
    - Rule-Based Word Segmentation
      - Example
        - Segment "Got a long list of ex-lovers, they'll tell you I'm insane (Yeah)"
        - What are the regular expression rule for the segmentation?
      - Example
        - themendinehere 
        - Possible Segmentation (Dictionary-Based)
      - Maximal Matching Algorithm  
    - Dictionary-Based 
    - Machine Learning System
    - Token
  - Sentence Segmentation
  - Morphological Analysis
- คำเป็นหน่วยทางทฤษฎี เป็นสิ่งที่ถูกสร้างขึ้นมา 

### Word Segmentation 

### Machine Learning

- Machine Learning is very accurate. 

### Subword Tokenization

- Token => หน่วยเล็กที่สุดที่อยากได้ / ตัวอักษรที่เกิดเห็นร่วมกันมากที่สุด 
- Three Common Algorithms
  - Byte-Pair Encoding (BPE)
  - Unigram Language Modeling Tokenization 
  - WordPiece
- Byte-Pair Encoding (BPE)
  - มี Corpus กับเซตของ Vocabulary (เป็นตัวอักษร)
  - หาว่าเจอชุดตัวอักษรไหนเยอะที่สุดใน Corpus
  - ลองเอามา Merge กัน  
  - เอามาใส่ในเซตของ Vocabulary 
  - Example
      - Corpus: low low low low low lowest lowest newer newer newer newer newer newer wider
wider wider new new  

### Sentence Segmentation 

Sentence
- เหนือกว่าวลี 
- หน่วยทางไวยากรณ์ (มีภาคประธาน + ภาคแสดง) 
- Sentence Boundary Detection 

### Morphological Analysis / Normalization 

- การวิเคราะห์หน่วยคำ 
- Morphology => หน่วยคำ 

## Week 02: Machine Learning for NLP 

### Traditional Technique
- Supervised Learning  
  - Learning from Label between Input and Output. 
- Unsupervised Learning 
  - Input such an Natural Grouping 
- Transfer Learning
  - Transfer จากแหล่งที่ไม่ใช้ Input / Output ชัดเจน ไปสู่จุดที่มี Input / Output 
  - Pre-Training Process 
  - Fine-Tuning Process 

### Topic Modeling 

### Text Classification 

### Other Example
- Automatic Essay Grading 
  - ใช้ในการสอบแบบ TOEFL, IELTS ที่จะมีการตรวจแบบออนไลน์  

### Type of Classifier
- Rule-Based
  - รับ Input -> เข้ากฏ -> ส่ง Output ตามกฏ 
  - เช่น เช็คอีเมล เช็คที่อยู่ 
    - ใช้ RegEx หรือ Lexicon ก็ได้ 
- Zero-shot LLM Classifier
  - Prompt แล้วทำให้กลายเป็น Classifier ได้เลย 
  - Zero-shot => ไม่ใช้ Training Data เลย
  - เช่น GPT, Gemini, etc.  
  - ไม่ต้องเก็บข้อมูล แต่ทำได้หลายอย่าง  
- ML-Based classifier
  - Popular Model
    - Logistic Regression 
    - Deep Learning 
  Steps
    - Data Prepare
      - เตรียมข้อมูล
    - Data Annotation
      - การกำกับข้อมูล 
      - กำหนด Label / Tag 
    - Feature Engineering 
      - เอา String / คำ มาแปลงเป็นค่าตัวเลข (ที่ยังเก็บความหมาย) เพื่อนำไปวิเคราะห์ 
      - Feature Vector   
      - Example 
        - Bag-of-Word Features 
    - Model Training 
      - Train-Validation-Test
        - Training Set => เอาไว้ฝึก  _
        - Validation Set => Development Set / Holdout Set
        - Test Set => เอาไว้ทดสอบ  
      - Deduplication
        - เอาตัวที่ซ้ำในแต่ละเซตออก  
      - Inference 
        - มี Text ใหม่มาเพื่อทำการอนุมาน 
    - Evaluation  
      - พัฒนาโมเดล
      - Example
        - Spam Classification 
      - Gold Standard => คำตอบจริง
      - Prediction => คำตอบจากการคาดการณ์  
      - Precision and Recall
        - Precision: Predict A ถูกกี่ครั้ง / จำนวนครั้งที่เรา Predict A 
          - เลือกแบบจุกจิก 
        - Recall: Predict A ถูกกี่ครั้ง / จำนวนครั้งที่ A ถูก  
          - เลือกแบบกวาดเข้ามา 
        - เอามาดูกับหลายตัวเลือก แล้วดูว่ามาตรวัดมันโอเคหรือไม่ 
      - F1 Score 
        - การเฉลี่ย Precision กับ Recall 
        - 2PR / (P + R)
      - Macro-Average 
        - เอา F1 Score แต่ละตัวมาเฉลี่ยเป็นค่าค่าเดียว 

## Week 03: Logistic Regression

- Logistic Regression
- Vector
  - ชุดของตัวเลขเรียงกัน แต่ใช้ตัวแปรตัวเดียวแทน 
- z = sum(weight_n * value_n) + bias  

### Binary Class  
- Sigmoid
  - Convert Z -> P(Y=1|X) 

### Multiclass
- Softmax: ค่าสูงสุด แบบไม่ฟันธง (Softๆ) 

### Matrix and Vector

- Dot Product: ถามว่าเวกเตอร์ 2 อัน คล้ายกันแค่ไหน  
- Cosine Similarity: การทำมุมกันของเวกเตอร์  

### Logarithm and Exponential

- ใช้ Log เพื่อเอาไปแปลงเลขให้ดูสวยขึ้น ให้ระบบเก็บเลขทศนิยมที่เยอะๆ 

### Log-Likelihood

- Logarithm ของ Prob ที่เราคำนวณมาได้
- สนใจแค่ Logarithm ของช่องที่ตรงกับ Label เท่านั้น  

### Cross-Entropy Loss

- ใช้ในเกือบทุกโมเดลที่เป็น Classification  

### Optimization Algorithm

- การหาค่าที่เหมาะสมที่สุด
- ใช้ Loss Function 
- ใช้ Gradient ในการบอกว่าควรจะปรับอย่างไรให้ Loss Function ออกมาดี 

### Calculus

- ใน NLP, X ไว้ใช้เป็น Parameter, F(X) เป็น Loss Function ที่เราต้องการที่จะ Minimize  

### Loss Function & Gradient Descent

- Gradient Descent เราใช้การกำหนดจุดแล้วให้ลงไปเรื่อยๆ ถ้าเริ่มถูกจะไปลงที่ Global Minimum ถ้าลงผิดจุดจะไปลงที่ Local Minimum  

### Stochastic Gradient Descent   

### Step to train data

- Read CSV
- Clean Data
- Feature Vector
- Split Train-Dev-Test
- Train on Training Set   
- Evaluate the dev set 
- Stratify
  - รันตอนไหนก็ต้องได้ผลลัพธ์ออกมาเหมือนกัน  


## Week 04: N-Gram Language Model

### Word Guessing
- หากเราคล่องภาษา เราจะสามารถเดาคำจากบริบทของประโยคได้ 
- ต้องมีความเข้าใจเกี่ยวกับการทำงานของธรรมชาติ หรือ ของโลกได้  

### Probability
- Probability Distribution over Vocabulary   
  - 0 < P(W=w) < 1
  - The sum of all probability distribution must be 1
  - We can sample a word from the distribution. Word with high probability must be frequently picked. 

### Bigram   

### Perplexity
- Lower = Better   


## Week 05: Deep Learning (I) - Word Embedding

### Lexical Semantics

### Lexical Relation  

### Computational Lexical Semantics  

###
เทรนรอบเดียวพอ เพราะเยอะ ใช้เวลานาน  


## Week 06: Deep Learning (II) - Neural Network   

### How the brain works?  

### Neural Network Unit

### Training Vocab
- Iteration: 
- Epoch: กี่ลูปในหนึ่งรอบการเทรนข้อมูล 

### Mini Batching

### Momentum
- [Playground](https://distill.pub/2017/momentum/)  

### Adam Optimizer  

## Week 07: Deep Learning (III) - GPT, BER

### GPT

- Self-Attention
  - ทำให้เข้าใจคำแบบที่เข้าใจบริบทของมันจริงๆ ไม่ใช้แค่เข้าใจเป็นความหมายตามพจนานุกรม 

### Encoder-Only Language Model

- Masked Language Model Loss
  - 
- BERT 
  - BERT Family
    - BERT (Google, 2018)
    - RoBERTa (Facebook, 2019)
    - DeBERTa (Microsoft, 2019)
  - Predict คนละ loss กับ GPT
    - GPT: ทายคำถัดไป
    - BERT: ทายคำตรงกลาง 
  - ไม่ค่อยมีอะไร ต้องไป Fine-Tuning ก่อน
  - Sequence Classification with BERT
    - CLS   
  - SOTA (State of the Art): สิ่งที่ดีที่สุดที่เคยเห็น หรือ เคยสังเกตในขณะนั้น  
- Multilingual Models 
  - Model ส่วนใหญ่เป็นภาษาอังกฤษ แต่ก็มีบางโมเดลที่ทำไว้เป็นภาษาอื่นด้วย  

## Week 08: Information Retrieval  
- Dot Product 
  - ดูว่า query กับ document มันแมทช์กันแค่ไหน
- Norm 
  - เป็นตัวหาร ทำให้ Document มีความเท่ากันนิดนึง 
- Cosine Score 

### Term Weighting with TF-IDF 
- มีข้อสันนิษฐานว่าคำที่เจอบ่อยๆทุกอัน แล้วเราเอาคะแนนของคำนั้นมานับ มันก็ไม่ดี เพราะทุกประโยคก็อาจจะเจอคำนั้นหมด สามารถตัดออกไปได้ (แต่เราอาจจะไม่ตัดออก แต่ใช้วิธีการให้ความสำคัญกับมันน้อยลง อีกอย่างคือมันแล้วแต่ document ด้วย)  
  - เช่น การรีวิวโรงแรม น่าจะมีคำว่า "โรงแรม" ในแต่ละรีวิว ดังนั้นเราจึงอาจจะให้ความสำคัญกับมันน้อยลง  
- Common Solution for Term Weighting
  - TF-IDF
    - Formula: weight = t_f * id_f  
      - Raw Count: tf_(t,d) = count(t, d)
      - Squash Way: tf_(t,d) = 1 + log(count(t,d), 10) if count(t,d) > 0; = 0 if otherwise;
        - log คือการบีบค่า ให้ค่าห่างกันไม่กว้างเกิน 
- Collection Frequency
  - เจอกี่ครั้งใน Collection 
- Document Frequency
  - เจอกี่ Document ใน Collection 
- Inverse Document Frequency (IDF)
  idf_t = log((N / df_t), 10); N -> Total Number of Documents in the collection 
  - df บอกความสำคัญของคำ

### Search in Practice
- การเลือกใช้สูตร เราจะเลือกใช้แบบไหนก็แล้วแต่การวางแผนของเราเลย
- Okapi BM25 (BM = Best Match)  
  - Formula: score(D, Q) = ...
  - การคำนวณแบบสูตร n(q_i) เป็นการคำนวณที่หลายๆเจ้าใช้  


### Efficiency: Inverted Index 
- เก็บดัชนี แทน Document ก็ได้ ทำให้กระชับขึ้น
- แทนที่จะเก็บแค่ Document เราจะเก็บด้วยว่า Term Frequency (tf) มีเท่าไหร่ 
- Assignment 02 อาจจะให้ทำเรื่อง Inverted Index 

### Evaluation of IR (Information Retrieval)
- มีการทำ Precision and Recall 
  - ดูความถูกใจเป็นหลัก อาจจะไม่ได้วัดความถูกต้องขนาดนั้น 
- Search ไม่เหมือนกับ Classifier คือ Data ถูก Annotated โดยอัตโนมัติ โดยจะบันทึกระหว่างที่ User ทำการเสิร์ช (มี Search Log มาให้) 
  - แต่อาจจะเกิดปัญหา Cold Start Problem คือในช่วงแรกจะไม่มีข้อมูล ทำให้ไม่รู้ว่าข้อมูลเป็นอย่างไร  

## Week 09: Information Retrieval (Continue)

### Evaluation of IR (Information Retrieval)

- Precision & Recall
  - Precision: % of selected items that are corrected.
  - Recall: % of corrected items that are selected.  

- Query, Doc_ID, Rank, Click, Precision@K
  - ดูแต่ละ Query ว่ามีตัวไหนใน Query ที่โดน Click (Click = 1)
  - ในแถวที่โดนคลิ๊ก ให้ดูว่าอยู่ใน Rank ไหน 
  - ให้หา Precision@K (Precision ของ Rank นั้น มาคิด)

### Conclusion
- Information Retrieval find relevant documents to address information needs.
- ใช้ TF-IDF ในการหาคำที่เกี่ยวข้อง  
- ใช้ MAP ในการ evaluate

## Week 09 (II): Advanced IR and RAG

### A/B Testing

- เปรียบเทียบแต่ละกลุ่ม ว่าทำแบบไหนให้ผลได้ดีกว่า หรือมีพฤติกรรมในการทำสิ่งนั้นๆในแต่ละกลุ่ม เหมือน หรือ ต่างกันอย่างไร  
- วัด Metrics อะไรก็ได้ 
- ตัวอย่าง
  - เปลี่ยนปุ่มแล้วมีผลต่อพฤติกรรมหรือไม่ อย่างไร 
- Step
  - Evaluate new system
  - Divert small amount of traffic
    - 1% new system + 99% status quo (status quo = สิ่งที่กำลังเป็นอยู่ตอนนี้)
  - Wait and Monitor
  - Roll out to more  
- Prop
  - ใช้การวัด Metrics แบบไหนก็ได้ 
  - วัดผลได้โดยตรงจากการใช้งานสิ่งที่เปลี่ยนไปจริงๆได้เลย 
- Cons
  - ใช้เวลานาน 
  - ต้องมี Infrastructure เบื้องหลังที่ดี ในการเปลี่ยนผ่านระบบไปสู่ผู้ใช้แต่ละคน 

### Query Understanding

- เปลี่ยนคำเป็น Token
  - Token ในที่นี้อาจจะ assume จากคำเลยก็ได้ เช่น
    - "พาสต้า โรแมนติก สีลม ไม่ แพง"
      - "สีลม" assume เป็น "Location: สีลม" (เห็นชื่อที่ดูเหมือนสถานที่ ก็เอาเข้าไปเป็นสถานที่)
      - "ไม่แพง" assume เป็น "Attribute: $$" หรือ "Price: $$" (เห็นคำที่ดูเหมือนแสดงความเห็นเรื่องราคา ก็เอาไปตีเป็นช่วงราคา)  
- Users don't tell everything.
  - Example
    - Distance
    - Rating
    - Open Now?  
    - New Restaurant 
- Features for search
  - TermScore(q, d) โดยใช้ TF-IDF
  - FromScore(q, d)
  - TitleScore(q, d)
  - QueryExpansionScore(q, d)  
  - Distance Score(u, d)
  - Recency Score(u, d) 