
# llm_recogn_entity .format(question)
rec_ent_prompt={
    "pubmed_qa":{
        "character": "You are a medical assistant to carry out Medical Name Entity Recognition from a question and output JSON like {{'entities': [entity1, entity2,...]}}.\
        Since the entity you extract is used to retrieve relevant information, you will generally pay more attention to rarer entities", 
        "request": """Get the meaningful entities in the question: [{}]. The answer should be like this:\
        Example1:\
        Q: The patient suffers from severe hydrophobia, you suspect that he has rabies and obtained a corneal scratch from the patient. What tests should this specimen be tested to diagnose rabies?\
        A:{{'entities': ['hydrophobia', 'rabies,corneal scratch']}} . \n
        Example2:\
        Q: What are the clinical manifestations of autologous and allogeneic CIK?\
        A:{{'entities': ['autologous CIK', 'allogeneic CIK']}} . \n
        Example3:\
        Q: The lateral spread of dental caries is facilitated mostly  by the\
        A:{{'entities': ['lateral spread of dental caries']}} . \n
        """
    },
    "BioASQ":{
        "character":"You are a Biological assistant to carry out Biological Name Entity Recognition from a question and output JSON like {{'entities': [entity1, entity2,...]}}.\
        Since the entity you extract is used to retrieve relevant information, you will generally pay more attention to rarer entities",
        "request":"""Get the meaningful entities in the question: [{}]. The answer should be like this:\
        Example1:\
        Q: Which is the most common editing modification in eukaryotic mRNA?\
        A:{{'entities': ['eukaryotic mRNA']}} . \n
        Example2:\
        Q: How many genes outside of the MHC locus have been genetically associated to Rheumatoid Arthritis through GWAS?\
        A:{{'entities': ['Rheumatoid Arthritis','GWAS']}} . \n
        """
    },
    "LawBenchmark":{
        "character":"你是一名法律助手，负责从问题中进行金融实体识别，并以JSON格式输出，格式为{{'entities': [entity1, entity2,...]}}。由于你提取的实体将用于检索相关信息，因此你通常会更加关注较为罕见的实体。",
        "request":"""获取问题中的有意义实体：[{}]。答案应如下所示:\n
        示例1:\
        Q: 关于贪污罪与挪用公款罪的犯罪主体，下列说法不正确的是\
        A:{{'entities': ['贪污罪的犯罪主体', '挪用公款罪的犯罪主体']}} . \n
        示例2:\
        Q: 根据我国现行宪法和有关法律规定、下列有关行政区域划分、行政区域边界争议处理的主管部门的表述中，哪一种说法是正确的?\
        A:{{'entities': ['行政区域划分', '行政区域边界争议处理']}} . \n
        """
    },
    "NQ-open":{
        "character":"You are a Polymath to carry out Natural Science Name Entity Recognition from a question and output JSON like {{'entities': [entity1, entity2,...]}}.\
        Since the entity you extract is used to retrieve relevant information, you will generally pay more attention to rarer entities",
        "request":"""Get the meaningful entities in the question: [{}]. The answer should be like this:\
        Example1:\
        Q: who is the designer in devil wears prada\
        A:{{'entities': ['devil wears prada']}} . \n
        Example2:\
        Q: who played bat masterson in the movie tombstone\
        A:{{'entities': ['bat masterson,tombstone']}} . \n
        """},
    "SimpleQA_science":{
        "character":"You are a Science assistant to carry out Science Name Entity Recognition from a question and output JSON like {{'entities': [entity1, entity2,...]}}.\
        Since the entity you extract is used to retrieve relevant information, you will generally pay more attention to rarer entities",
        "request":"""Get the meaningful entities in the question: [{}]. The answer should be like this:\
        Example1:\
        Q: Which is the most common editing modification in eukaryotic mRNA?\
        A:{{'entities': ['eukaryotic mRNA']}} . \n
        Example2:\
        Q: How many genes outside of the MHC locus have been genetically associated to Rheumatoid Arthritis through GWAS?\
        A:{{'entities': ['Rheumatoid Arthritis','GWAS']}} . \n
        """
    },
}

# llm_relation_score .format(question,str(retrieve_result))
rel_score_prompt ={
    "pubmed_qa":{
        "character": "You are an excellent medical assistant with extensive and accurate medical knowledge. You are asked to evaluate some triplets with validity scores to help your doctor answer questions. The range of validity scores is between 1.0 and 0.0. If you think a triple is more helpful in answering this question, the higher it scores, and vice versa. A score of 1.0 means that the triple can answer the question directly, and as the score decreases (0.95, 0.9, 0.8...) the triple helps to answer questions less. You are asked to output your results in json format",
        "request":"Based on the question: [{q}], score the triples: [{t}]. \n \
        The score is between 0 and 1 and the order is from high to low. Here is an example:\n \
        Q: What is the cause of hereditary Parkinson's disease?, triples: {{head: Parkinson, relation:is, tail: doctor}}, {{head: SNCA missense mutation, relation:may cause, tail: Hereditary Parkinson's disease}},{{head: Parkinson's disease, relation:symptoms, tail: Tremor}},{{head: Hereditary Parkinson's disease, relation:causes, tail: SNCA missense mutation}} \n \
        A: {{triples:[{{triple:{{head: Hereditary Parkinson's disease, relation:causes, tail: SNCA missense mutation}}, score:1.0}}, {{triple:{{head: SNCA missense mutation, relation:may cause, tail: Hereditary Parkinson's disease}}, score:0.85}},{{triple:{{head: Parkinson's disease, relation:symptoms, tail: Tremor}}, score:0.3}},{{triple:{{head: Parkinson, relation:is, tail: doctor}}, score:0.0}}]}}"
    },
    "BioASQ":{
        "character": "You are an excellent biology teaching assistant with extensive and accurate biological knowledge. You are asked to evaluate some triplets with validity scores to help your professor answer questions. The range of validity scores is between 1.0 and 0.0. If you think a triple is more helpful in answering this question, the higher it scores, and vice versa. A score of 1.0 means that the triple can answer the question directly, and as the score decreases (0.95, 0.9, 0.8...) the triple helps to answer questions less. You are asked to output your results in JSON format",
        "request": "Based on the question: [{q}], score the triples: [{t}]. \
        The score is between 0 and 1 and the order is from high to low. Here is an example:\
        Q: What is the main function of mitochondria in animal cells?, triples: {{head: Mitochondria, relation: produce, tail: ATP}}, {{head: Mitochondria, relation: are found in, tail: plant cells}}, {{head: Mitochondria, relation: contain, tail: DNA}},{{head: Mitochondria, relation: contains, tail: ATP-producing enzymes}},{{head: Altman, relation: discovered, tail: mitochondria}} \
        A: {{triples:[{{triple:{{head: Mitochondria, relation: produce, tail: ATP}}, score:1.0}},{{triple:{{head: Mitochondria, relation: contains, tail: ATP-producing enzymes}}, score:0.8}},{{triple:{{head: Mitochondria, relation: are found in, tail: plant cells}}, score:0.4}},{{triple:{{head: Mitochondria, relation: contain, tail: DNA}}, score:0.2}},{{triple:{{head: Altman, relation: discovered, tail: mitochondria}}, score:0.0}}]}}"
    },
    "LawBenchmark":{
        "character": "你是一位出色的法学助教，拥有广泛且准确的法律知识。现在需要你对一些三元组进行有效性评分，以帮助你的教授回答问题。有效性评分的范围为1.0到0.0。如果你认为某个三元组对回答该问题更有帮助，则给出的分数应越高，反之则分数越低。得分为1.0表示该三元组可以直接回答该问题，分数越低（如0.95、0.9、0.8等），则说明该三元组对回答问题的帮助程度越低。请将你的结果以JSON格式输出。",
        "request": "基于问题: [{q}], 给下列的三元组打分: [{t}]. \
        打出的分数应该在0到1之间，并且从高到低排序。下面是一个示例： \
        Q: 公司是否可以因为员工怀孕而解除劳动合同？, triples: {{head: 劳动合同法, relation: 禁止, tail: 因女职工怀孕解除劳动合同}}, {{head: 员工怀孕, relation: 属于, tail: 特殊保护期}}, {{head: 解除劳动合同, relation: 需要, tail: 提前三十日通知员工}},{{head: 劳动合同法, relation: 颁布时间, tail: 2007年6月29日}} \
        A: {{triples:[{{triple:{{head: 劳动合同法, relation: 禁止, tail: 因女职工怀孕解除劳动合同}}, score:1.0}},{{triple:{{head: 员工怀孕, relation: 属于, tail: 特殊保护期}}, score:0.9}},{{triple:{{head: 解除劳动合同, relation: 需要, tail: 提前三十日通知员工}}, score:0.3}},{{triple:{{head: 劳动合同法, relation: 颁布时间, tail: 2007年6月29日}}, score:0.0}}]}}"
    },
    "NQ-open":{
        "character": "You are an excellent Polymath with extensive and accurate knowledge. You are asked to evaluate some triplets with validity scores to help you answer questions. The range of validity scores is between 1.0 and 0.0. If you think a triple is more helpful in answering this question, the higher it scores, and vice versa. A score of 1.0 means that the triple can answer the question directly, and as the score decreases (0.95, 0.9, 0.8...) the triple helps to answer questions less. You are asked to output your results in JSON format",
        "request": "Based on the question: [{q}], score the triples: [{t}]. \
        The score is between 0 and 1 and the order is from high to low. Here is an example:\
        Q: What causes the seasons on Earth?, triples: {{head: Seasons, relation: caused by, tail: Earth's axial tilt}}, {{head: Earth, relation: tilted axis, tail: 23.5 degrees}}, {{head: Earth, relation: revolves around, tail: Sun}}, {{head: Day and night, relation: caused by, tail: Earth's rotation}} \
        A: {{triples:[{{triple:{{head: Seasons, relation: caused by, tail: Earth's axial tilt}}, score:1.0}},{{triple:{{head: Earth, relation: tilted axis, tail: 23.5 degrees}}, score:0.7}},{{triple:{{head: Earth, relation: revolves around, tail: Sun}}, score:0.6}},{{triple:{{head: Day and night, relation: caused by, tail: Earth's rotation}}, score:0.0}}]}}"
    },
    "SimpleQA_science":{
        "character": "You are an excellent science teaching assistant with extensive and accurate science knowledge. You are asked to evaluate some triplets with validity scores to help your professor answer questions. The range of validity scores is between 1.0 and 0.0. If you think a triple is more helpful in answering this question, the higher it scores, and vice versa. A score of 1.0 means that the triple can answer the question directly, and as the score decreases (0.95, 0.9, 0.8...) the triple helps to answer questions less. You are asked to output your results in JSON format",
        "request": "Based on the question: [{q}], score the triples: [{t}]. \
        The score is between 0 and 1 and the order is from high to low. Here is an example:\
        Q: What is the main function of mitochondria in animal cells?, triples: {{head: Mitochondria, relation: produce, tail: ATP}}, {{head: Mitochondria, relation: are found in, tail: plant cells}}, {{head: Mitochondria, relation: contain, tail: DNA}},{{head: Mitochondria, relation: contains, tail: ATP-producing enzymes}},{{head: Altman, relation: discovered, tail: mitochondria}} \
        A: {{triples:[{{triple:{{head: Mitochondria, relation: produce, tail: ATP}}, score:1.0}},{{triple:{{head: Mitochondria, relation: contains, tail: ATP-producing enzymes}}, score:0.8}},{{triple:{{head: Mitochondria, relation: are found in, tail: plant cells}}, score:0.4}},{{triple:{{head: Mitochondria, relation: contain, tail: DNA}}, score:0.2}},{{triple:{{head: Altman, relation: discovered, tail: mitochondria}}, score:0.0}}]}}"
    },
}


create_triple_prompt ={
    "pubmed_qa":{
        "character":"""You are a medical expert to Extract knowledge from Q&A and Create Knowledge Graph Triples just based on the information given without your own knowledge designed to output JSON.The triples you extract will be used in different questions, so you should pay more attention to extracting some general knowledge than special cases in this question.""", 
        "request":"""Based on the question: [{}] and answer: [{}], first think about the general knowledge contained in this question and answer,\
        and don't output these knowledge and change the case of the letter, then use these knowledge to generate the most meaningful knowledge graph triples such as: {{triples: [<triple1>, <triple2>, ...]}},\
        each triple should be like: {{head: xxx, relation: xxx, tail: xxx}}.\
        Here is an example for your reference:
        Example1:\
        Question: Can a patient with asthma use beta-blockers?\
        Answer: no.\
        Your answer: {{triples: [{{"head": "asthma", "relation": "contraindicated_with", "tail": "beta-blockers"}}]}}
        """
    },
    "BioASQ":{
        "character":"""You are a Biological expert to Extract knowledge from Q&A and Create Knowledge Graph Triples just based on the information given without your own knowledge designed to output JSON.The triples you extract will be used in different questions, so you should pay more attention to extracting some general knowledge than special cases in this question.""",
        "request":"""Based on the question: [{}] and answer: [{}], first think about the general knowledge contained in this question and answer,\
        and don't output these knowledge and change the case of the letter, then use these knowledge to generate the most meaningful knowledge graph triples such as: {{triples: [<triple1>, <triple2>, ...]}},\
        each triple should be like: {{head: xxx, relation: xxx, tail: xxx}}.\
        Here is an example for your reference:\n
        Example1:\
        Question: If a patient with asthma uses a beta-blocker, what might happen\n
        Answer: Beta-blockers can worsen asthma symptoms, so they should be used with caution or avoided in asthmatic patients.\n
        Your answer: {{triples: [{{"head": "beta-blockers", "relation": "can_worsen", "tail": "asthma symptoms"}},{{"head": "asthma", "relation": "contraindicated_with", "tail": "beta-blockers"}}]}}
        """
    },
    "LawBenchmark":{
        "character":"你是一位法律专家，负责从问答中抽取知识，并仅根据所给信息创建知识图谱三元组，不要使用你自己的知识，输出格式为JSON。这些你抽取的三元组将用于解答不同的问题，因此你应更加关注抽取一般性知识，而不是本题中的个例。",
        "request":"""根据问题：[{}] 和答案：[{}]，首先思考该问答中包含的一般性知识，不要输出这些知识，而是使用这些知识生成最有意义的知识图谱三元组，格式如下：{{triples: [<triple1>, <triple2>, ...]}}，\
                    每个三元组的格式应为：{{head: xxx, relation: xxx, tail: xxx}}。\
                    以下是一个供你参考的示例：\n
                    示例一：\
                    问题：下列哪种情形可以采取简易程序进行处理？\n
                    答案：被告人是未成年人\n
                    你的回答：{{triples: [{{"head": "被告人是未成年人", "relation": "可采取", "tail": "简易诉讼程序"}}]}}\n
                    示例二：\
                    问题：下面关于宪法和国际条约的关系，说法不正确的有:\n
                    答案：国务院拥有批准和废除同外国缔结条约的职权
                    你的回答：{{triples: [{{"head": "国务院", "relation": "没有", "tail": "批准和废除同外国缔结条约的职权"}}]}}\n"""
    },
    "NQ-open":{
        "character":"""You are a Polymath to Extract knowledge from Q&A and Create Knowledge Graph Triples just based on the information given without your own knowledge designed to output JSON.The triples you extract will be used in different questions, so you should pay more attention to extracting some general knowledge than special cases in this question.""",
        "request":"""Based on the question: [{}] and answer: [{}], first think about the general knowledge contained in this question and answer,\
        and don't output these knowledge and change the case of the letter, then use these knowledge to generate the most meaningful knowledge graph triples such as: {{triples: [<triple1>, <triple2>, ...]}},\
        each triple should be like: {{head: xxx, relation: xxx, tail: xxx}}.\
        Here is an example for your reference:
        Example1:\
        Question: If a patient with asthma uses a beta-blocker, what might happen?\n
        Answer: Beta-blockers can worsen asthma symptoms, so they should be used with caution or avoided in asthmatic patients.\n
        Your answer: {{triples: [{{"head": "beta-blockers", "relation": "can_worsen", "tail": "asthma symptoms"}},{{"head": "asthma", "relation": "contraindicated_with", "tail": "beta-blockers"}}]}}
        """
    },
    "SimpleQA_science":{
        "character":"""You are a Science expert to Extract knowledge from Q&A and Create Knowledge Graph Triples just based on the information given without your own knowledge designed to output JSON.The triples you extract will be used in different questions, so you should pay more attention to extracting some general knowledge than special cases in this question.""",
        "request":"""Based on the question: [{}] and answer: [{}], first think about the general knowledge contained in this question and answer(Please pay more attention to the answer and the information in the question that is related to the answer.),\
        and don't output these knowledge and change the case of the letter, then use these knowledge to generate the most meaningful knowledge graph triples such as: {{triples: [<triple1>, <triple2>, ...]}},\
        each triple should be like: {{head: xxx, relation: xxx, tail: xxx}}.\
        Here is an example for your reference:\n
        Example1:\
        Question: What two academic disciplines did Nancy Amato earn bachelor's degrees in at Stanford University in 1986?\n
        Answer: Mathematical Sciences and Economics\n
        Your answer: {{triples: [{{"head": "Nancy Amato", "relation": "earned_bachelor's_degree_in", "tail": "Mathematical Sciences and Economics"}},{{"head": "Nancy Amato", "relation": "earned_bachelor's_degree_from", "tail": "Stanford University"}},{{"head": "Nancy Amato", "relation": "earned_bachelor's_degree_in_year", "tail": "1986"}}]}}
        """,
    },
}

datasets_prompt = {

    "pubmed_qa":{
        "character":"""You are a medical expert to answer the question and output JSON. Your answer must be one of the following: [yes / no / maybe / unknown]. The output must be a dict with only two keys: answer and support_info and your output must be like {{'answer':xxx,'support_info':xxx}}.""",
        "request":"""Based on your own knowledge and the <knowledge triple>(if available; otherwise, ignore it), choose one of the [yes / no / maybe / unknown] to answer the question. \n
        Q: [{q}], knowledge triple: [{i}], A:?\n
        Here is some examples for your reference:\n
        Example1: \
        Q: Is Sesto ed Uniti located in the Central Standard Time zone? \
        Knoledage triple:[{{'triple': {{'head': 'sesto ed uniti', 'relation': 'time zone', 'tail': 'Central European Time Zone'}},'score': 0.95}},{{'triple': {{'head': 'sesto ed uniti', 'relation': 'locate in', 'tail': 'Italy'}},'score': 0.9}},{{'triple': {{'head': 'China', 'relation': 'capital', 'tail': 'beijing'}}, 'score': 0}}]
        A: {{'answer':'no', \n "support_info":[{{'triple': {{'head': 'sesto ed uniti', 'relation': 'time zone', 'tail': 'Central European Time Zone'}},'score': 0.95}},{{'triple': {{'head': 'sesto ed uniti', 'relation': 'locate in', 'tail': 'Italy'}},'score': 0.9}}]}}
        Example2: \
        Q: When a patient's body temperature reaches 39 degrees Celsius, does he have a fever? \
        Knoledage triple:["no related knowledge"]
        A: {{'answer':'yes', \n "support_info":["Based my own knowledge and common sense, a body temperature above 38°C is considered a symptom of fever, so think the answer is yes"]}}
        """,
    },
    "BioASQ":{
        "character":"""You are a biologist. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only two keys: answer and support_info and your output must be like {{'answer':xxx,'support_info':xxx}}.""",
        "request":"""Based on your own knowledge and the <knowledge triple>, please answer the question concisely.\n
        Q: [{q}], knowledge triple: [{i}], A:?\n
        Example: \
        Q: What disease can be treated with Glofitamab? \
        Knoledage triple:[{{'triple': {{'head': 'Glofitamab', 'relation': 'treats', 'tail': 'DLBCL'}}, 'score': 1.0}},{{'triple': {{'head': 'Glofitamab', 'relation': 'is a', 'tail': 'monoclonal antibody'}}, 'score': 0.6}}]
        A: {{'answer':'DLBCL'，\n "support_info":[{{'triple': {{'head': 'Glofitamab', 'relation': 'treats', 'tail': 'DLBCL'}}, 'score': 1.0}}]}}""",
    },
    "LawBenchmark":{
        "character":"""你是一名法学教授，请你回答下列问题，并以 JSON 格式输出结果。你的答案必须为 [\"0\"/\"1\"/\"2\"/\"3\"] 中的一个数字，表示你的选择顺序。输出必须是一个只包含两个键（answer和support_info）的字典，格式如下：{{'answer':xxx, 'support_info':xxx}}。""",
        "request":"""基于你自身的知识，并结合<knowledge triple>中提供的知识，选择下面的一个选项以回答问题。\n
        Q: [{q}], knowledge triple: [{i}], Option:[{c}], A:?\n
        例子：\n
        Q: 秦朝把杀伤、盗窃等危害封建统治的犯罪称为()。\n
        knowledge triple: [{{'triple': {{'head': '杀伤、盗窃', 'relation': '在秦朝被称为', 'tail': '公室告'}}, 'score': 1.0}}, {{'triple': {{'head': '公室告', 'relation': '包括', 'tail': '杀人'}}, 'score': 0.7}}]\n
        Option:["公罪","私罪","公室告","非公室告"]\n
        A: {{'answer':\"2\", 'support_info':[{{'triple': {{'triple': {{'head': '杀伤、盗窃', 'relation': '在秦朝被称为', 'tail': '公室告'}}, 'score': 1.0}}]}}
 """,
    },
    "NQ-open":{
        "character":"""You are a Polymath. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only two keys: answer and support_info and your output must be like {{'answer':xxx,'support_info':xxx}}.""",
        "request":"""Based on your own knowledge and the <knowledge triple>, please answer the question concisely.\n
        Q: [{q}], knowledge triple: [{i}], A:?\n
        Example: \
        Q: what is the meaning of the name comanche? \
        Knoledage triple:[{{'triple': {{'head': 'comanche', 'relation': 'means', 'tail': 'enemy'}},'score':1.0}},{{'triple':{{'head': 'comanche', 'relation': 'means', 'tail': 'enemy'}},'score':0.7}},{{'triple':{{'head': 'The term 'Comanche'', 'relation': 'comes from ', 'tail': 'Ute word 'kɨmantsi''}},'score':0.7}}]
        A: {{'answer':'enemy'，\n "support_info":[{{'triple': {{'head': 'comanche', 'relation': 'means', 'tail': 'enemy'}},'score':1.0}}]}}""",
    },
    "SimpleQA_science":{
        "character":"""You are a science expert. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only two keys: answer and support_info and your output must be like {{'answer':xxx,'support_info':xxx}}.""",
        "request":"""Based on your own knowledge and the <knowledge triple>, please answer the question concisely.\n
        Q: [{q}], knowledge triple: [{i}], A:?\n
        Example: \
        Q: What disease can be treated with Glofitamab? \
        Knoledage triple:[{{'triple': {{'head': 'Glofitamab', 'relation': 'treats', 'tail': 'DLBCL'}}, 'score': 1.0}},{{'triple': {{'head': 'Glofitamab', 'relation': 'is a', 'tail': 'monoclonal antibody'}}, 'score': 0.6}}]
        A: {{'answer':'DLBCL'，\n "support_info":[{{'triple': {{'head': 'Glofitamab', 'relation': 'treats', 'tail': 'DLBCL'}}, 'score': 1.0}}]}}""",
    },
}



Temporary_prompt ={
    "pubmed_qa":{
        "character": "You are a medical expert to answer the question and output JSON. Your answer must be one of the following: [yes / no / maybe / unknown]. The output must be a dict with only one keys: answer and your output must be like {{'answer':xxx}}.",
        "request": "Based on your own knowledge, choose one of the [yes / no / maybe / unknown] to answer the question. \
        Q: [{q}],  A:?\
        Example: \
        Q: Are sugars-free medicines more erosive than sugars-containing medicines? \
        A: {{'answer':'no'}}"},
    "BioASQ":{
        "character":"You are a biologist. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only one keys: answer and your output must be like {{'answer':xxx}}.",
        "request":"Based on your own knowledge, please answer the question concisely.  \
        Q: [{q}],  A:?\
        Example: \
        Q: What disease can be treated with Glofitamab? \
        A: {{'answer':'DLBCL'}}"
    },
    "LawBenchmark":{
        "character": "你是一名律师，请你回答下列问题，并以 JSON 格式输出结果。你的答案必须为 [\"0\"/\"1\"/\"2\"/\"3\"] 中的一个数字，表示你的选择顺序。输出必须是一个仅包含 answer 这一键的字典，格式如下：{{'answer': xxx}}。",
        "request": """基于你自身的知识, 选择下面的一个选项以回答问题. \
        Q: [{q}], Option:[{c}], A:?\
        例子: \
        Q: 赔偿请求人向共同赔偿义务机关中的一个赔偿义务机关要求赔偿的，该赔偿义务机关应如何办理?, Option:["该赔偿义务机关仅就自己所应承担责任的范围内予以赔偿","该赔偿义务机关有权拒绝赔偿","该赔偿义务机关可以就赔偿总额先予赔偿","该赔偿义务机关应当就赔偿总额先予赔偿"] \
        A: {{'answer':\"3\"}}""",
    },
    "NQ-open":{
        "character":"You are a Polymath. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only one keys: answer and your output must be like {{'answer':xxx}}.",
        "request":"Based on your own knowledge, please answer the question concisely.  \
        Q: [{q}],  A:?\
        Example: \
        Q: what is the meaning of the name comanche? \
        A: {{'answer':'enemy'}}"
    },
    "SimpleQA_science":{
        "character":"You are a Scientist. Please provide concise and accurate answers to the following questions.The output must be a dict in JSON format with only one keys: answer and your output must be like {{'answer':xxx}}.",
        "request":"Based on your own knowledge, please answer the question concisely.  \
        Q: [{q}],  A:?\
        Example: \
        Q: What disease can be treated with Glofitamab? \
        A: {{'answer':'DLBCL'}}"
    },
}



# 聚类提示词
clustering_example = """
{{
    "clusters": [
        {{
            "1": [
                "Paris.",
                "Paris.",
                "It's Paris"
            ]
        }},
        {{
            "2": [
                "Rome."
            ]
        }},
        {{
            "3": [
                "New York"
            ]
        }}
    ]
}}
"""

choice_clustering_example = """
{{
    "clusters": [
        {{
            "1": [
                "1",
                "1",
                "1"
            ]
        }},
        {{
            "2": [
                "2",
            ]
        }},
        {{
            "3": [
                "3",
            ]
        }}
    ]
}}
"""

clustering_prompt = {
    "system": """ You are a logician. Here we define a relation called "entail". Suppose there are two sentences A and B. If A can logically imply B (proving B to be true), then we say A entails B. If A and B mutually entail each other, then A and B can be considered semantically identical and classified as the same category. For example, the sentences "The capital of France is Paris" and "Paris is the capital of France" mutually entail each other, meaning these two sentences are equivalent and can be considered as belonging to the same category.""",
    "user": """ Now, I will provide you with some sentences, they are the answers for the question: {q}. Please cluster the sentences provided below and return the results in JSON format (note: please return directly in JSON format without outputting any other content!!!): {sentences}; Below is a reference example of the return result (only for style reference, the content is unrelated to your task)""" + "\n" + clustering_example
}

choice_cluster_prompt = {
    "system": """ You are a logician. Here we define a relation called "entail". Suppose there are two sentences A and B. If A can logically imply B (proving B to be true), then we say A entails B. If A and B mutually entail each other, then A and B can be considered semantically identical and classified as the same category. For example, the sentences "The capital of France is Paris" and "Paris is the capital of France" mutually entail each other, meaning these two sentences are equivalent and can be considered as belonging to the same category.""",
    "user": """ Now, I will provide you with some numbers. Please cluster the numbers provided below and return the results in JSON format (note: please return directly in JSON format without outputting any other content!!!): {sentences}; Below is a reference example of the return result (only for style reference, the content is unrelated to your task)""" + "\n" + choice_clustering_example
}



insufficient_info_prompt = """ I will provide you with some classified responses from large language models. Please identify which categories represent responses indicating 'insufficient information' or 'unable to answer'(just mean that the model cannot answer the question), and return the corresponding category numbers in JSON format, for example {{'insufficient_info':[1,2,3]}}. If none of them indicate insufficient information, please return {{'insufficient_info':[]}}.You should attention that only the information about the insufficient_info cluster is needed, other information(like your reason) is not needed.If the answer is just a number(like "1" or "2"), this group is ignored."""


insufficient_info_prompt_pub = """ I will provide you with some classified responses from large language models. Please identify which categories represent responses indicating 'insufficient information' or 'unable to answer'(just mean that the model cannot answer the question), and return the corresponding category numbers in JSON format, for example {{'insufficient_info':[1,2,3]}}. If none of them indicate insufficient information, please return {{'insufficient_info':[]}}.You should attention that only the information about the insufficient_info cluster is needed, other information(like your reason) is not needed.If the answer is just a number(like "1" or "2"), this group is ignored."""


confidence_judgment_prompt = {
    "LawBenchmark":{
        "character":"""你是一名法学专家，拥有广泛且准确的法律知识。  
根据你自身的知识以及下方提供的知识三元组，你认为你有可能能够回答下面的多项选择题吗（题目和选项会在下方给出）？请一步一步思考你应该如何作答。最后，请用JSON格式回答Yes或No，格式如下：{"answer"："Yes"} 或 {"answer"："No"}。""",
        "request":"""问题：{q}\n选项：{c}\n知识：{i}\n你的回答：""",
        },
}

answer_judege_prompt = """
I will provide you with a question, a answer generated by llm and a real_answer. Please judge whether the llm_answer is correct.
You just need to return a JSON with one key: 'result', and the value is a boolean value.
If the llm_answer is correct, the result should be true, otherwise it should be false.

QUESTION[{q}]
LLM_ANSWER[{llm_answer}]
REAL_ANSWER[{real_answer}]"""
