## longformer_test.py

###
# Portions of this script are sourced from the article "Transformers for Long
# Text, Code Examples with Longformer" by Christian Versloot.  His work can be
# accessed here:
# https://github.com/christianversloot/machine-learning-articles/blob/main/transformers-for-long-text-code-examples-with-longformer.md
###

# Import tensorflow, transformer models from huggingface, and dependencies
import numpy as np
import tensorflow as tf
import torch
from transformers import LongformerTokenizer, TFEncoderDecoderModel

# Get and print GPU info to ensure GPU usage
gpus = tf.config.list_physical_devices('GPU')
gpu_info = tf.config.experimental.get_device_details(gpus[0])
print('Device name: ', gpu_info['device_name'])

# Load model and tokenizer
model = TFEncoderDecoderModel.from_pretrained(
    "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16"
    ,from_pt = True
)

tokenizer = LongformerTokenizer.from_pretrained(
    "allenai/longformer-base-4096"
)

# Provide example article/survey response
germany_article = """
Germany (German: Deutschland, officially 
the Federal Republic of Germany, is a country at the intersection of Central 
and Western Europe. It is situated between the Baltic and North seas to the 
north, and the Alps to the south; covering an area of 357,022 square kilometres 
(137,847 sq mi), with a population of over 83 million within its 16 
constituent states. It borders Denmark to the north, Poland and the Czech 
Republic to the east, Austria and Switzerland to the south, and France, 
Luxembourg, Belgium, and the Netherlands to the west. Germany is the 
second-most populous country in Europe after Russia, as well as the most 
populous member state of the European Union. Its capital and largest city is 
Berlin, and its financial centre is Frankfurt; the largest urban area is the 
Ruhr. Various Germanic tribes have inhabited the northern parts of modern 
Germany since classical antiquity. A region named Germania was documented 
before AD 100. In the 10th century, German territories formed a central part 
of the Holy Roman Empire. During the 16th century, northern German regions 
became the centre of the Protestant Reformation. Following the Napoleonic Wars 
and the dissolution of the Holy Roman Empire in 1806, the German Confederation 
was formed in 1815. In 1871, Germany became a nation-state when most of the 
German states unified into the Prussian-dominated German Empire. After World 
War I and the German Revolution of 1918???1919, the Empire was replaced by the 
semi-presidential Weimar Republic. The Nazi seizure of power in 1933 led to the 
establishment of a dictatorship, World War II, and the Holocaust. After the end 
of World War II in Europe and a period of Allied occupation, Germany was 
divided into the Federal Republic of Germany, generally known as West Germany, 
and the German Democratic Republic, East Germany. The Federal Republic of 
Germany was a founding member of the European Economic Community and the 
European Union, while the German Democratic Republic was a communist Eastern 
Bloc state and member of the Warsaw Pact. After the fall of communism, German 
reunification saw the former East German states join the Federal Republic of 
Germany on 3 October 1990???becoming a federal parliamentary republic led by a 
chancellor.Germany is a great power with a strong economy; it has the largest 
economy in Europe, the world's fourth-largest economy by nominal GDP, and the 
fifth-largest by PPP. As a global leader in several industrial, scientific and 
technological sectors, it is both the world's third-largest exporter and 
importer of goods. As a developed country, which ranks very high on the Human 
Development Index, it offers social security and a universal health care 
system, environmental protections, and a tuition-free university education. 
Germany is also a member of the United Nations, NATO, the G7, the G20, and the 
OECD. It also has the fourth-greatest number of UNESCO World Heritage Sites.
"""

# Provide survey results (sourced from NNL's Glassdoor page)
survey_pro = '''
Everyone who works at NNL, either out of school or coming from another company 
really takes to the camaraderie and sense of community at NNL. For many at NNL, 
it is a labor of love and folks here have a genuine pride in their work and 
that shows in the way people here interact with each other. NNL is also 
generous in allowing employees to move between technical areas and gain a 
breadth of experiences. Medical benefits are excellent.
'''

survey_con = '''
This is a company rife with ineffective operations. Purely technical people 
without business experience and without other industry exposure operate in an 
environment lacking real consequences for poor performance. As they are the 
sole provider of reactor design and fuel disposal to the U.S. Navy, the only 
heat management feels is when they lay out a plan and fail to achieve the plan. 
This is not to say that the original plan ever had merit. And the consequence 
is merely a tongue lashing from Navy oversight. Age discrimination is rife. 
The homogeneous (white, men) foundation of the program is retiring, which is 
creating a massive talent loss that cannot be directly replaced. Years of 
mismanaging talent has led to an equally poor solution - promote people in 
their 20s and 30s to management. The employee base now has a 'double-hump' 
with the experienced class retiring and the inexperienced class making poor 
decisions across the board. The highly insecure and political environment has 
a never ending stream of ineffective results.
'''

# combine the two responses to test how the model blends them in a summary
survey_combo = survey_pro + ' ' + survey_con

# Tokenize and Summarize
input_ids = tokenizer(survey_con, return_tensors = 'pt').input_ids
output_ids = model.generate(input_ids)

# Get the summary from the output tokens
summary = tokenizer.decode(output_ids[0], skip_special_tokens = True)

# Print summary
print('\n', summary)
