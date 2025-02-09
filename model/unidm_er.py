#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import pandas as pd

from utils.chain_of_thoughts_module import get_chain_of_thoughts_template,get_chain_of_thoughts_template_eval
from model.unidm_base import UniDM
from utils.constants import MATCH_PROD_NAME


class UniDM_EntityResolution(UniDM):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.dataset_name = args.data_dir.split('/')[-1]
        prod_name = MATCH_PROD_NAME[self.dataset_name]
        # mine
        self.prompt_dp = f"Given this {prod_name} and some meta data for the {prod_name}. Can you retrieve the specific model (sku) of {prod_name}, please be concise? \n The {prod_name} is %s"
        self.prompt_dp = f"Given the items and convert the them into a textual format in a logical order.\n The items are %s.\n The {prod_name} is "
        self.pe_suffix = f"Do {prod_name} A and {prod_name} B describe the same entity? Yes or No. "
        self.template = f"The {prod_name} A is %s The {prod_name} B is %s"
        self.context = ""
        self.prod_name=prod_name
        self.num_self_consistency = args.num_self_consistency
        self.use_chain_of_thoughts=args.use_chain_of_thoughts

    def instance_retrieval(self, train):
        """
        The Instance-wise component of the auto-retrieve module.
        """
        # data balance
        labels = train['label_str'].unique()
        instances = [train['label_str'] == l for l in labels]
        print("let's see before pd.concat ")
        print(instances)
        instances = pd.concat([ins.sample(self.context_num) for ins in instances])
        print("let's see before train.sample ")
        print(instances)
        instances = train.sample(self.instance_num)
        print("let's see after train.sample ")
        print(instances)

        context = ""
        for i,row in instances.iterrows():
            entity_A, entity_B = row["serialized_A"],row["serialized_B"]
            if self.Data_Parsing:
                entity_A, entity_B = self.data_parsing(row["serialized_A"]), self.data_parsing(row["serialized_B"])
            pre, gt = self.pe_suffix, row["label_str"].strip()
            if self.use_chain_of_thoughts:
                context_r = get_chain_of_thoughts_template(row,instances.columns.tolist(),self.prod_name) +"\n"+ self.template % (entity_A, entity_B) +f"{pre} {gt}" 
            else:
                context_r = self.template % (entity_A, entity_B) +f"{pre} {gt}" 

            context += context_r + "\n\n"
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
        if self.use_chain_of_thoughts:
            prompt_pe = self.context +get_chain_of_thoughts_template_eval(row,columns_list,self.prod_name) +query
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
            pred = self.apply_prompt(prompt=prompt_pe)
                # print(pred)
            
            preds.append(pred)
            gt = row["label_str"].strip()
            self.logger.info(f"idx:{i} ====> pred:{pred} / gt:{gt}")
        
        return preds
            