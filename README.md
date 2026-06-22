# Loss Functions, Explained

Small, correct, tested versions of the loss functions that come up in medical
imaging, with a note on when each one earns its place.

Focal loss is for heavy class imbalance, where rare positives get drowned out by
easy negatives. Dice loss measures overlap and holds up when the foreground is a
tiny fraction of the image, which is most of segmentation. Tversky loss lets you
tune the trade between false positives and false negatives, so when missing a
finding is worse than a false alarm you raise beta to push recall. NT-Xent is the
contrastive loss behind SimCLR, for pretraining without labels.

## Tests

```
pip install -r requirements.txt
pytest tests/ -q
```

These are property tests, not just "does it run". Each loss comes out lower when
the prediction is right. Tversky collapses back to Dice when alpha and beta are
both 0.5. Raising Tversky's beta makes it punish missed positives harder.
NT-Xent is lower when the positive pairs actually match. If the math is wrong,
the tests fail.
