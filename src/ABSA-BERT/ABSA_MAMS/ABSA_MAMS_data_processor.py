import xml.etree.ElementTree as ET
import json
import spacy

nlp = spacy.load('en_core_web_lg')

train_path = '../../../data/mams_atsa_train.xml'
train_absa_path = 'ABSA_MAMS_train.json'

if __name__ == "__main__":
    output = []
    
    xml_data = open(train_path, 'r').read()
    root = ET.XML(xml_data)

    data = []
    for _, sentence in enumerate(root):
        opinions = []
        text = ''

        opinions = []

        for _, sentence_children in enumerate(sentence):
            if sentence_children.tag == 'aspectTerms':
                for _, opinion in enumerate(sentence_children):
                    opinions.append({
                        'target': opinion.attrib['term'],
                        'polarity': opinion.attrib['polarity'],
                        'from': int(opinion.attrib['from']),
                        'to': int(opinion.attrib['to']),
                    })
            elif sentence_children.tag == 'text':
                text = sentence_children.text

        doc = nlp(text)

        iob = []
        absa_tags = []

        opinion = None

        tokens = []

        for token in doc:
            if opinion == None:
                for op in opinions:
                    if token.idx >= op['from'] and token.idx + len(token.text) <= op['to']:
                        opinion = op
                        break
                if opinion == None:
                    iob.append(0)
                else:
                    iob.append(1)
            else:
                if token.idx >= opinion['from'] and token.idx + len(token.text) <= opinion['to']:
                    iob.append(2)
                else:
                    iob.append(0)
                    opinion = None
            
            labsa = len(absa_tags)

            for op in opinions:
                if token.idx >= op['from'] and token.idx + len(token.text) <= op['to']:
                    if op['polarity'] == 'positive':
                        absa_tags.append(3)
                    elif op['polarity'] == 'negative':
                        absa_tags.append(1)
                    elif op['polarity'] == 'neutral':
                        absa_tags.append(2)
                    
                    break
            
            
            if labsa == len(absa_tags):
                absa_tags.append(0)
            
            tokens.append(token.text)

        for opinion in opinions:
            data.append({
                'text': text,
                'tokens': tokens,
                'absa_tags': absa_tags,
            })
    
    with open(train_absa_path, 'w') as f:
        json.dump(data, f)