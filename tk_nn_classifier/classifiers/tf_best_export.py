import tensorflow as tf
import tensorflow.logging as logging


class BestCheckpointsExporter(tf.estimator.BestExporter):

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        if self._best_eval_result is None or \
                self._best_eval_result['accuracy'] < eval_result['accuracy']:
            logging.info(
                'Exporting a better model ({} instead of {})...'.format(
                    eval_result, self._best_eval_result))
            result = self._saved_model_exporter.export(
                estimator, export_path, checkpoint_path, eval_result,
                is_the_final_export)
            self._best_eval_result = eval_result
            self._garbage_collect_exports(export_path)
            return result
        else:
            logging.info(
                'Keeping the current best model ({} instead of {}).'.format(
                    self._best_eval_result, eval_result))
