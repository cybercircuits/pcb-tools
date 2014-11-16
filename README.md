pcb-tools
====================================================================

This is a fork of [pcb-tools](https://github.com/curtacircuitos/pcb-tools) modified to perform a single function:
clip a Gerber layer against the bounding box of another layer. I have used
this to clip silkscreen layers so they don't extend beyond the edges of the PCB
(which can increase fabrication costs because the manufacturer thinks the board
is larger than it actually is).

To clip a layer, use the following command:

    python clip.py -i silk.GTO -c board.GKO -o clipped_silk.GTO 

where:

* `silk.GTO` is the layer to be clipped (usually a silkscreen layer).
* `board.GKO` is the layer whose bounding box will be used for clipping (usually the board outline).
* `clipped_silk.GTO` is the resulting layer after the clipping operation.

This program has not been extensively tested. **Always check the result using a Gerber viewer!**
