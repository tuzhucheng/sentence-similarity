import os
import subprocess
import uuid

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
import torch


class RetrievalMetrics(Metric):
    """
    Calculates retrieval metrics using trec_eval
    `update` must receive output of the form (ids, y_pred, y).
    """
    def reset(self):
        self._ids = []
        self._predictions = []
        self._gold = []

    def update(self, output):
        ids, y_pred, y = output
        self._ids.extend(ids)
        self._predictions.append(y_pred)
        self._gold.append(y)

    def compute(self):
        if len(self._predictions) == 0:
            raise NotComputableError('MAP/MRR must have at least one example before it can be computed')

        predicted_scores = torch.cat(self._predictions).data.cpu().numpy()
        gold_scores = torch.cat(self._gold).data.cpu().numpy()

        randid = uuid.uuid4()
        qrel_fname = '{}.qrel'.format(randid)
        results_fname = '{}.results'.format(randid)
        qrel_template = '{qid} 0 {docno} {rel}\n'
        results_template = '{qid} 0 {docno} 0 {sim} mymodel\n'

        docnos = range(len(self._ids))
        zipped_lines = list(zip(self._ids, docnos, predicted_scores, gold_scores))
        zipped_lines.sort(key=lambda t: t[0])
        relevant_doc_exists = set()
        for line in zipped_lines:
            if line[3] > 0:
                relevant_doc_exists.add(line[0])

        with open(qrel_fname, 'w') as f1, open(results_fname, 'w') as f2:
            for qid, docno, predicted, actual in zipped_lines:
                if qid in relevant_doc_exists:
                    f1.write(qrel_template.format(qid=qid, docno=docno, rel=actual))
                    f2.write(results_template.format(qid=qid, docno=docno, sim=predicted))

        trec_eval_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trec_eval-9.0.5/trec_eval')
        trec_out = subprocess.check_output([trec_eval_path, '-m', 'map', '-m', 'recip_rank', qrel_fname, results_fname])
        trec_out_lines = str(trec_out, 'utf-8').split('\n')
        mean_average_precision = float(trec_out_lines[0].split('\t')[-1])
        mean_reciprocal_rank = float(trec_out_lines[1].split('\t')[-1])

        os.remove(qrel_fname)
        os.remove(results_fname)

        return {'map': mean_average_precision, 'mrr': mean_reciprocal_rank}


class MAP(RetrievalMetrics):
    """
    Calculates the MAP.
    `update` must receive output of the form (ids, y_pred, y).
    """
    def compute(self):
        retrieval_metrics = super(MAP, self).compute()
        return retrieval_metrics['map']


class MRR(RetrievalMetrics):
    """
    Calculates the MRR.
    `update` must receive output of the form (ids, y_pred, y).
    """
    def compute(self):
        retrieval_metrics = super(MRR, self).compute()
        return retrieval_metrics['mrr']
