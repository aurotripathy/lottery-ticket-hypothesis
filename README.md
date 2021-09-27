# lottery-ticket-hypothesis

    (pyt1.2) auro@auro-ml:~/simple-pruning-example$ python3 main.py 
    CUDA enabled.
    Preparing: Train before prune...
    Train epoch: 0, Val accuracy: 0.4
    Train epoch: 1, Val accuracy: 0.4997
    Train epoch: 2, Val accuracy: 0.5893
    Train epoch: 3, Val accuracy: 0.6766
    Train epoch: 4, Val accuracy: 0.7545
    Train epoch: 5, Val accuracy: 0.7937
    Train epoch: 6, Val accuracy: 0.82
    Train epoch: 7, Val accuracy: 0.8336
    Train epoch: 8, Val accuracy: 0.8482
    Train epoch: 9, Val accuracy: 0.8576
    Train epoch: 10, Val accuracy: 0.8658
    Train epoch: 11, Val accuracy: 0.8696
    Train epoch: 12, Val accuracy: 0.8752
    Train epoch: 13, Val accuracy: 0.8795
    Train epoch: 14, Val accuracy: 0.8837

    Pruning...
    Layer 1 | Linear layer | 78.88% parameters pruned
    Layer 2 | Linear layer | 46.48% parameters pruned
    Layer 3 | Linear layer | 18.40% parameters pruned
    Current pruning rate: 74.88%

    Train after prune...
    Train epoch: 0, Val accuracy: 0.8082
    Train epoch: 1, Val accuracy: 0.8364
    Train epoch: 2, Val accuracy: 0.8537
    Train epoch: 3, Val accuracy: 0.8602
    Train epoch: 4, Val accuracy: 0.863
    Train epoch: 5, Val accuracy: 0.8688
    Train epoch: 6, Val accuracy: 0.8705
    Train epoch: 7, Val accuracy: 0.8726
    Train epoch: 8, Val accuracy: 0.8755
    Train epoch: 9, Val accuracy: 0.8756
    Train epoch: 10, Val accuracy: 0.8778
    Train epoch: 11, Val accuracy: 0.8786
    Train epoch: 12, Val accuracy: 0.881
    Train epoch: 13, Val accuracy: 0.8824
    Train epoch: 14, Val accuracy: 0.8841
    --- After retraining ---
    Test accuracy: 0.8841

