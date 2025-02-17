#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import pandas as pd

from utils.data_utils import clean_text, count_words
from utils.chain_of_thoughts_module import get_chain_of_thoughts_template,get_chain_of_thoughts_template_eval
from model.unidm_base import UniDM
from utils.constants import MATCH_PROD_NAME
import spacy
import re
# PARSING_OPTIONS = ['Yes\n','No\n', "Can't decide\n"]
CONFIRM = "Yes\n"
DENY = "No\n"
ABSTAIN = "Can't decide\n"
NAN_TOKEN="nan"
IGNORED_COLUMNS= ["ltable_id","rtable_id","id","id_A","id_B","label_str","label","serialized_A","serialized_B"]
THRESHOLD_COUNT_WORDS =5
nlp = spacy.load("en_core_web_lg")

feature_toggle=True

class UniDM_EntityResolution(UniDM):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.dataset_name = args.data_dir.split('/')[-1]
        prod_name = MATCH_PROD_NAME[self.dataset_name]
        self.clean_text=args.clean_text
        # mine
        # self.prompt_dp = f"Given this {prod_name} and some meta data for the {prod_name}. Can you retrieve the specific model (sku) of {prod_name}, please be concise? \n The {prod_name} is %s"
        self.prompt_dp = f"Given the items and convert the them into a textual format in a logical order.\n The items are %s.\n The {prod_name} is "
        self.pe_suffix = f"Do {prod_name} A and {prod_name} B describe the same entity? Yes or No. "
        self.template = f"The {prod_name} A is %s The {prod_name} B is %s"
        self.context = ""
        self.prod_name=prod_name
        self.num_self_consistency = args.num_self_consistency
        self.use_chain_of_thoughts_llm=args.use_chain_of_thoughts_llm
        self.use_chain_of_thoughts_spacy = args.use_chain_of_thoughts_spacy
        self.use_chain_of_thoughts_experimental=args.use_chain_of_thoughts_experimental
        self.Data_Parsing =  not( self.use_chain_of_thoughts_experimental or self.use_chain_of_thoughts_spacy  or self.use_chain_of_thoughts_llm) and args.data_parsing
        if feature_toggle:
            self.context_map={}
        if self.use_chain_of_thoughts_experimental:
            self.pe_suffix = f"Considering Key Identifiers, Core Specifications and Units of Measurements. Do {prod_name} A and {prod_name} B describe the same entity? Yes or No. "
            self.template = """
**Product 1:**  
%s \n
**Product 2:**  
%s \n
                            """
            self.template_context = """
**Product 1:**  
%s \n
**Product 2:**  
%s \n
%s \n
                            """
                            # - Any synonyms or variations in wording
                            # - Packaging & Quantity (Single item vs. Bundles)  
            self.pe_suffix = f"""You are an expert in product matching and entity resolution. Your task is to determine whether two product descriptions refer to the same product.

### **Instructions:**
To decide whether two products are the same, follow these steps:
1. **Identify Key Attributes:** Compare **brand, model, specifications, and unique features**.  
2. **Analyze Technical Details:** Check **storage, RAM, display size, and additional features**.  
3. **Compare Functionality & Category:** Ensure both products serve the **same purpose**.  
4. **Detect Variations or Ambiguities:** Consider **synonyms and missing details** before making a decision.  


### **Question:**  
Do these two product descriptions refer to the same product?  
Respond with one of the following:
- "YES" if they are the same product.
- "NO" if they are different products.

                            """
    def are_texts_similar(self,text1, text2,label=None, spacy_threshold=0.6,att="description"):
        if label  is not None :
            return label,label==CONFIRM
        if self.clean_text:
            text1 = clean_text(text1)
            text2 = clean_text(text2)
        text1_passed_throughLLM=text1
        text2_passed_throughLLM=text2
        print(f"text1 is:: {text1}")
        print(f"text2 is:: {text2}")
        # use spacy if we have too little words
        if count_words(text1)<=THRESHOLD_COUNT_WORDS and  count_words(text2)<=THRESHOLD_COUNT_WORDS:
            doc1, doc2 = nlp(text1), nlp(text2)
            if feature_toggle:
                doc1, doc2 = nlp(text1.lower()), nlp(text2.lower())
            similarity = doc1.similarity(doc2)  # Compute similarity score
            print(f"similarity spacy is {similarity}")
            # use spacy to get threshold and return always yes if they are matching already
            return (CONFIRM,1) if  similarity >= spacy_threshold else (DENY,0)
        if not feature_toggle:
            # ask the LLM
            template_summarization =f"Given the following {att}, summarize it in no more than 10 words without removing important details such as make and model and highlight key specs of the {self.prod_name}.\n The {att} is %s.\n The summary is "
            # if count_words(text1)>=THRESHOLD_COUNT_WORDS:
            #     text1_passed_throughLLM =self.apply_prompt_dp(template_summarization % text1)
            #     print(f"text1 after LLM is:: {text1_passed_throughLLM}")
            # if count_words(text2)>=THRESHOLD_COUNT_WORDS:
            #     text2_passed_throughLLM =self.apply_prompt_dp(template_summarization % text2)
            #     print(f"text2 after LLM is:: {text2_passed_throughLLM}")
            text1_passed_throughLLM=self.apply_prompt_dp(template_summarization % text1_passed_throughLLM)
            text2_passed_throughLLM=self.apply_prompt_dp(template_summarization % text2_passed_throughLLM)
            if self.Data_Parsing:
                text1_passed_throughLLM=self.data_parsing(text1_passed_throughLLM)
                text2_passed_throughLLM=self.data_parsing(text2_passed_throughLLM)
            print(f"text1 after parsing is:: {text1_passed_throughLLM}")
            print(f"text2 after parsing is:: {text2_passed_throughLLM}")
        template_similarity=f"Product A's {att} is %s.\n  Product B's {att} is %s. If different languages are used please translate the different language to English. Do these {att}s representing same {self.prod_name}? Yes or No."
        template_similarity=f"Product A's {att} is %s.\n  Product B's {att} is %s. Do these {att}s representing same {self.prod_name}? Yes or No."
        prompt_similarity = template_similarity % (text1_passed_throughLLM,text2_passed_throughLLM)
        response =self.apply_prompt(self.context+ '\n '+prompt_similarity)
        response = response.strip().lower().split("\n")[0]
        
        print(f"Response for similarity is {response}")
        response = CONFIRM if "yes" in response else DENY
        print(f"similarity llm is {response}")
        # return response
        # doc1, doc2 = nlp(text1), nlp(text2)
        # similarity = doc1.similarity(doc2)  # Compute similarity score
        # print(f"similarity spacy is {similarity}")
        # use spacy to get threshold and return always yes if they are matching already
        return (CONFIRM,1) if response in CONFIRM  else (DENY,0)

    def get_chain_of_thoughts_attribute_segregation(self,row,columns_list,provide_analysis=False):
        columns_for_matching=   {c.split("_")[0] for c in columns_list if not (c in IGNORED_COLUMNS) }
        # print("ROW IS::")
        # print(row)
        label = row["label_str"]
        att_seg_A=f"\n"
        att_seg_B=f"\n"
        matching_prompt_template = f"""The following two %s are the same and represent the same {self.prod_name}.\n
        Product 1 %s: %s \n
        Product 2 %s: %s \n
        Can you provide a short description in no more than 10 words why they represent the same {self.prod_name}?"""
        not_matching_prompt_template = f"""The following two %s are different and represent different {self.prod_name}.\n
        Product 1 %s: %s \n
        Product 2 %s: %s \n
        Can you provide a short description in no more than 10 words why they represent the different {self.prod_name}?"""
        analysis= "**Step-by-Step Analysis:**\n"
        for col in columns_for_matching:
            if pd.isna(row[col+"_A"]) or pd.isna(row[col+"_B"]) or "null" in str(row[col+"_B"]).lower() or "null" in str(row[col+"_A"]).lower() :
                continue
            att_seg_A+=f"-{col}: {clean_text(row[col+'_A'])}\n"
            att_seg_B+=f"-{col}: {clean_text(row[col+'_B'])}\n"
            att_seg_B+=f"\n"
            if provide_analysis:
                response_llm_dp = self.apply_prompt_dp((matching_prompt_template if label==CONFIRM else not_matching_prompt_template)%(col,col,row[col+"_A"],col,row[col+"_B"]))
                analysis += response_llm_dp + "\n"
            # chain_of_thoughts+= f"regarding '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
            # answer,bool_answer = self.are_texts_similar(row[col+"_A"],row[col+"_B"],label=label,att=col)
            # cumulative_answers+=bool_answer
            # queried_answers+=1
            # chain_of_thoughts += are_these_attributes_similar % (col, answer )
        if not provide_analysis:
            return att_seg_A,att_seg_B
        return att_seg_A,att_seg_B,analysis
        
        # return chain_of_thoughts, CONFIRM if cumulative_answers == queried_answers else DENY
        
    def get_chain_of_thoughts_template(self,row,columns_list, prod_name):
        columns_for_matching=   {c.split("_")[0] for c in columns_list if not (c in IGNORED_COLUMNS) }
        # print("ROW IS::")
        # print(row)
        label = row["label_str"]
        chain_of_thoughts=f"You are an expert in {prod_name} entity resolution. Given two product data, analyze their attributes step by step and determine if they refer to the same product.\n"
        if feature_toggle:
            chain_of_thoughts=f"You are an expert in {prod_name} entity resolution. Given two {prod_name} data, analyze their attributes step by step and determine if they refer to the same {prod_name}.\n"
            chain_of_thoughts=f""
        are_these_attributes_similar= f"Are %s similar for such {prod_name}s? Yes or No. %s"
        are_these_attributes_similar= f"Are these %s representing same entity? Yes or No. %s"
        cumulative_answers =0
        queried_answers =0
        for col in columns_for_matching:
            if pd.isna(row[col+"_A"]) or pd.isna(row[col+"_B"]) or "null" in str(row[col+"_B"]).lower() or "null" in str(row[col+"_A"]).lower() :
                continue
            if feature_toggle:
                chain_of_thoughts+= f"for the attribute '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
            else:
                chain_of_thoughts+= f"regarding '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
                
            answer,bool_answer = self.are_texts_similar(row[col+"_A"],row[col+"_B"],label=label,att=col)
            cumulative_answers+=bool_answer
            queried_answers+=1
            chain_of_thoughts += are_these_attributes_similar % (col, answer )
        return chain_of_thoughts, CONFIRM if cumulative_answers == queried_answers else DENY

    def get_chain_of_thoughts_template_eval(self,row,columns_list, prod_name):
        columns_for_matching=   {c.split("_")[0] for c in columns_list if not (c in IGNORED_COLUMNS) }
        # print("ROW IS::")
        # print(row)
        # label = row["label_str"]
        chain_of_thoughts=f"Now, Given two {prod_name} data, analyze their attributes step by step and determine if they refer to the same product.\n"
        if feature_toggle:
            chain_of_thoughts=f"Now, Given two {prod_name} data, analyze their attributes step by step and determine if they refer to the same {prod_name}.\n"
        
        are_these_attributes_similar= f"Are %s similar for such {prod_name}s? Yes or No. %s"
        are_these_attributes_similar= f"Are these %s representing same entity? Yes or No. %s"
        cumulative_answers =0
        queried_answers =0
        for col in columns_for_matching:
            if pd.isna(row[col+"_A"]) or pd.isna(row[col+"_B"]) or "null" in str(row[col+"_B"]).lower() or "null" in str(row[col+"_A"]).lower() :
                continue
            if feature_toggle:
                chain_of_thoughts+= f"for the attribute '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
            else:
                chain_of_thoughts+= f"regarding '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
            answer,bool_answer=self.are_texts_similar(row[col+"_A"],row[col+"_B"],label=None,att=col)
            cumulative_answers+=bool_answer
            queried_answers+=1
            chain_of_thoughts += are_these_attributes_similar % (col, answer )
        return chain_of_thoughts, CONFIRM if cumulative_answers == queried_answers else DENY
    def instance_retrieval(self, train):
        """
        The Instance-wise component of the auto-retrieve module.
        """
        # data balance
        labels = train['label_str'].unique()
        instances = [train['label_str'] == l for l in labels]
        instances = pd.concat([ins.sample(self.context_num) for ins in instances])
        instances = train.sample(self.instance_num,random_state=self.seed)
        counter=1
        context = ""
        if feature_toggle:
            context=f"Given two {self.prod_name} data, analyze their attributes step by step and determine if they refer to the same {self.prod_name}.\n"
        for i,row in instances.iterrows():
            entity_A, entity_B = row["serialized_A"],row["serialized_B"]
            
            if self.Data_Parsing:
                entity_A, entity_B = self.data_parsing(row["serialized_A"]), self.data_parsing(row["serialized_B"])
            pre, gt = self.pe_suffix, row["label_str"].strip()
            if self.use_chain_of_thoughts_llm:
                if feature_toggle:
                    cot,_ = self.get_chain_of_thoughts_template(row,instances.columns.tolist(),self.prod_name)
                    context_r = cot +"\n"+ self.template % (entity_A, entity_B) +f"\n {pre} {gt}" 
                    context_r = cot +"\n"+ f"\n {pre} {gt}" 
                else:
                    cot,_ = self.get_chain_of_thoughts_template(row,instances.columns.tolist(),self.prod_name)
                    context_r = cot +"\n"+ self.template % (entity_A, entity_B) +f"\n {pre} {gt}" 

            elif self.use_chain_of_thoughts_spacy:
                context_r = get_chain_of_thoughts_template(row,instances.columns.tolist(),self.prod_name) +"\n"+ self.template % (entity_A, entity_B) +f"\n {pre} {gt}" 
            elif self.use_chain_of_thoughts_experimental:
                context_r = self.template_context %  self.get_chain_of_thoughts_attribute_segregation(row,instances.columns.tolist(),provide_analysis=True) +f"\n {pre} {gt}" 

            else:
                context_r = self.template % (entity_A, entity_B) +f"\n {pre} {gt}" 
            if self.use_chain_of_thoughts_experimental:
                context += f"**Example {counter}:** \n"+ context_r + "\n\n"
                counter+=1
            else:
                context += f"**Example {counter}:** \n"+ context_r + "\n\n"
                counter+=1
        self.context = context
        print("Our context is :: ")
        print(self.context)

    def data_parsing(self, context):
        """
        Adaptive data parsing module.
        """
        prompt = self.prompt_dp % context
        # prompt = self.prompt_dp_mine % context
        gen_text = self.apply_prompt(prompt=prompt)
        # output = gen_text.strip('\n')
        # time.sleep(TIMESLEEP)
        return gen_text

    def prompt_engineering(self, target,row,columns_list):
        """
        Prompt engineering module.
        :param target: The target row.
        """
        entity_A, entity_B = target
        pre = self.pe_suffix
        query = self.template % (entity_A, entity_B) + f"{pre}"
        if self.use_chain_of_thoughts_llm:
            if feature_toggle:
                cot,answer =  self.get_chain_of_thoughts_template_eval(row,columns_list,self.prod_name)
                prompt_pe = self.context + cot +query
                prompt_pe = self.context + cot +f"{pre}"
                return prompt_pe,answer
                x=1
            else:
                cot,answer =  self.get_chain_of_thoughts_template_eval(row,columns_list,self.prod_name)
                prompt_pe = self.context + cot +query
                return prompt_pe,answer
        elif self.use_chain_of_thoughts_spacy:
            prompt_pe =  self.context + get_chain_of_thoughts_template_eval(row,columns_list,self.prod_name) +query 
        elif self.use_chain_of_thoughts_experimental:
            prompt_pe = self.context +self.template%self.get_chain_of_thoughts_attribute_segregation(row,columns_list) +f'{pre}'
        else:
            prompt_pe = self.context +query
        return prompt_pe

    def run(self, train_data, test_data):
        """
        :param train_data: The dataset to get the context.
        :param test_data: The dataset to test.
        """
        if self.instance_wise:
            self.instance_retrieval(train_data)

        preds = []
        for i,row in test_data.iterrows():
            entity_A, entity_B = row["serialized_A"],row["serialized_B"]
            if self.Data_Parsing:
                entity_A, entity_B = self.data_parsing(row["serialized_A"]), self.data_parsing(row["serialized_B"])
            
            # entity_A, entity_B = self.data_parsing(row["serialized_A"]), self.data_parsing(row["serialized_B"])
            prompt_pe = self.prompt_engineering([entity_A, entity_B],row,test_data.columns.tolist())
            self.p_as.append(prompt_pe)
            # for i in range(self.num_self_consistency):
            if self.use_chain_of_thoughts_llm:
                if feature_toggle:
                   pred= self.apply_prompt(prompt=prompt_pe[0])
                else:
                    pred=prompt_pe[1]
            else:
                pred = self.apply_prompt(prompt=prompt_pe)
                # print(pred)
            
            preds.append(pred)
            gt = row["label_str"].strip()
            self.logger.info(f"idx:{i} ====> pred:{pred} / gt:{gt}")
        
        return preds
            