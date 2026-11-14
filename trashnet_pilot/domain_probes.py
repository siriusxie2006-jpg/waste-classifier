"""Content-preserving interventions on a canonical 224-pixel model frame."""
import numpy as np
from PIL import Image

MEAN_RGB = (124, 116, 104)
CONDITIONS = ['original_white', 'existing_mean_pad', 'shrink75_white',
              'shrink75_mean', 'rotate90', 'rotate180']


def probe_frame(frame, condition, content_box):
    if frame.size != (224, 224) or condition not in CONDITIONS:
        raise ValueError('Expected canonical frame and a predefined condition')
    if condition == 'original_white':
        return frame.copy()
    if condition == 'existing_mean_pad':
        left, top, right, bottom = map(int, content_box)
        if not 0 <= left < right <= 224 or not 0 <= top < bottom <= 224:
            raise ValueError('Invalid preserved-content box')
        result = Image.new('RGB', (224, 224), MEAN_RGB)
        result.paste(frame.crop((left, top, right, bottom)), (left, top))
        return result
    if condition.startswith('shrink75_'):
        result = Image.new('RGB', (224, 224), 'white' if condition.endswith('white') else MEAN_RGB)
        result.paste(frame.resize((168, 168), Image.Resampling.BILINEAR), (28, 28))
        return result
    return frame.transpose(Image.Transpose.ROTATE_90 if condition == 'rotate90' else Image.Transpose.ROTATE_180)


def agreement_metrics(truth, scores, reference):
    truth, scores, reference = np.asarray(truth), np.asarray(scores), np.asarray(reference)
    prediction, original = scores.argmax(1), reference.argmax(1)
    midpoint = (scores+reference)/2
    epsilon = 1e-12
    js = .5*(scores*np.log((scores+epsilon)/(midpoint+epsilon))).sum(1)
    js += .5*(reference*np.log((reference+epsilon)/(midpoint+epsilon))).sum(1)
    return dict(prediction_agreement=float((prediction == original).mean()),
                original_correct_now_wrong=int(((original == truth) & (prediction != truth)).sum()),
                original_wrong_now_correct=int(((original != truth) & (prediction == truth)).sum()),
                mean_js_divergence=float(js.mean()))
