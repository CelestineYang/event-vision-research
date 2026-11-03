"""Guard model selection, prediction coverage and portable result exports."""
import unittest
from export_main_ablation_results import check_predictions, portable, select


class ResultExportTests(unittest.TestCase):
    def rows(self):
        return [dict(epoch=e, evaluation={'main':dict(video_top1=a,video_nll=n)})
                for e,a,n in [(1,.8,.4),(2,.9,.6),(3,.9,.5),(4,.9,.5)]]

    def test_top1_ties_and_nll_are_distinct(self):
        self.assertEqual(select(self.rows(),'video_top1_then_nll')['epoch'],3)
        self.assertEqual(select(self.rows(),'video_nll')['epoch'],1)

    def test_incomplete_or_unknown_selection_rejected(self):
        for rows in ([],[{'epoch':1,'train':{'video_top1':1}}]):
            with self.assertRaises(ValueError):select(rows,'video_top1_then_nll')
        with self.assertRaises(ValueError):select(self.rows(),'unreviewed')

    def test_prediction_score_and_coverage(self):
        predictions=[dict(video_id='a',label=0,prediction=0,windows=3),
                     dict(video_id='b',label=1,prediction=0,windows=2)]
        metrics=dict(videos=2,windows=5,video_top1=.5)
        self.assertEqual(check_predictions(predictions,metrics),1)
        for changed in (dict(metrics,windows=4),dict(metrics,videos=3),dict(metrics,video_top1=1)):
            with self.assertRaises(ValueError):check_predictions(predictions,changed)
        with self.assertRaises(ValueError):
            check_predictions([predictions[0],predictions[0]],metrics)

    def test_paths_and_runtime_metadata_removed(self):
        source={'path':'/machine/project/store','weights':['/another/private/best.pt'],
                'environment':{'credentials':'never export'},'nested':{'pid':123,'seed':0}}
        actual=portable(source,[('/machine/project','PROJECT_ROOT')])
        self.assertEqual(actual,{'path':'PROJECT_ROOT/store','weights':['LOCAL_PATH/best.pt'],
                                 'nested':{'seed':0}})


if __name__=='__main__':unittest.main()
