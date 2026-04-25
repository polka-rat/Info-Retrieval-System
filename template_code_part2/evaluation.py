from util import *

# Add your import statements here
import math

class Evaluation():

	def _build_qrels_dict(self, qrels):
		qrels_dict = {}
		for item in qrels:
			qid = int(item["query_num"])
			docid = int(item["id"])
			relevance = int(item["position"])
			
			if qid not in qrels_dict:
				qrels_dict[qid] = []
				
			qrels_dict[qid].append([docid,relevance])
			
		
		return qrels_dict

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here

		count =0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			if query_doc_IDs_ordered[i] in true_doc_IDs :
				count+=1
		
		precision = count/k if k > 0 else 0

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanPrecision = -1

		#Fill in code here
		n=len(query_ids)
		precisionsum=0
		qrels_dict = self._build_qrels_dict(qrels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			precisionsum+=self.queryPrecision(doc_IDs_ordered[i],query_ids[i],true_docs, k)
		
		meanPrecision=precisionsum/n

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		recall = -1

		#Fill in code here

		count =0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			if query_doc_IDs_ordered[i] in true_doc_IDs :
				count +=1
		if len(true_doc_IDs) == 0:
			return 0
		
		recall=count/(len(true_doc_IDs))
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanRecall = -1

		#Fill in code here
		n=len(query_ids)
		recallsum=0
		qrels_dict = self._build_qrels_dict(qrels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			recallsum+=self.queryRecall(doc_IDs_ordered[i],query_ids[i],true_docs,k)
		
		meanRecall=recallsum/n

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		fscore = -1

		#Fill in code here
		precision=self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
		recall = self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
		if precision + recall ==0 :
			return 0
		fscore = 1.25 / ((1/precision) + (0.25/recall))
		
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanFscore = -1

		#Fill in code here
		n=len(query_ids)
		fscoresum=0
		qrels_dict = self._build_qrels_dict(qrels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			fscoresum+=self.queryFscore(doc_IDs_ordered[i],query_ids[i], true_docs,k)
		
		meanFscore=fscoresum/n

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		nDCG = -1

		#Fill in code here
		
		qrels_dict = self._build_qrels_dict(qrels)
		rel_order={}
		for docid, rel in qrels_dict.get(query_id,[]) :
			rel_order[docid]=rel
		DCG =0.0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			doc_id = query_doc_IDs_ordered[i]
			relevance=rel_order.get(doc_id,0)
			DCG += relevance / math.log2(i + 2)
		
		sorted_rel = sorted(rel_order.values(), reverse=True)
        
		IDCG = 0.0
		for i in range(min(k, len(sorted_rel))):
			IDCG +=  sorted_rel[i]/ math.log2(i + 2)
		if IDCG == 0:
			return 0
		nDCG = DCG / IDCG
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanNDCG = -1

		#Fill in code here
		n = len(query_ids)
		total = 0
		qrels_dict = self._build_qrels_dict(qrels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			total += self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrels, k)

		meanNDCG=total/n
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		avgPrecision = -1

		#Fill in code here

		count =0
		precision_sum=[]
		for i in range(min(k, len(query_doc_IDs_ordered))):
			if query_doc_IDs_ordered[i] in true_doc_IDs :
				count+=1
				precision_sum.append(count/(i+1))
		
		if len(precision_sum)==0:
			return 0
		avgPrecision=sum(precision_sum)/len(precision_sum)
		

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		meanAveragePrecision = -1

		#Fill in code here
		n=len(query_ids)
		avgprecisionsum=0
		qrels_dict = self._build_qrels_dict(q_rels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			avgprecisionsum+=self.queryAveragePrecision(doc_IDs_ordered[i],query_ids[i],true_docs,k)
		
		meanAveragePrecision=avgprecisionsum/n

		return meanAveragePrecision



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		reciprocalRank = -1

		#Fill in code here
		for i in range(min(k, len(query_doc_IDs_ordered))) :
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				reciprocalRank=1/(i+1)
				break
		else :
			reciprocalRank=0

		return reciprocalRank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		meanReciprocalRank = -1

		#Fill in code here
		
		n=len(query_ids)
		reciprocalranksum=0
		qrels_dict = self._build_qrels_dict(qrels)
		for i in range(n):
			true_docs = [doc for doc, rel in qrels_dict.get(query_ids[i], []) if rel >= 1]
			reciprocalranksum+=self.queryReciprocalRank(doc_IDs_ordered[i],query_ids[i],true_docs,k)

		meanReciprocalRank=reciprocalranksum/n

		return meanReciprocalRank
