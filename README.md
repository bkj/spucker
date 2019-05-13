### spucker

This is a from-scratch reimplementation of [TuckER](https://github.com/ibalazevic/TuckER/).

#### Differences from original implementation
 - more efficient data loaders
 - more efficient validation phase
 - use `BCEWithLogitsLoss(x, y)` instead of `BCELoss(torch.sigmoid(x), y))` to improve numerical stability
 - don't materialize dense target matrix

#### Usage

See `./run.sh` for usage.
