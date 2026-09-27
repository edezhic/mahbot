## Leftover processes

A command that exits while a process it started still holds the command's output channel is reported as a FINISHED call — its own exit status and the output above it are the result. The process it left behind is left RUNNING, no further output from it is collected, and it is named where the containment it lives in can be named; where it cannot, the note says exactly that. {{mode_sentence}}

---

[leftover process] the command exited; its own exit status and the output above are its result. A process it left behind still holds its output channel, so no more of that output can be collected. The leftover is left running — mahbot holds that channel open so a closed pipe cannot end it, keeping the 16 most recent such channels and releasing the oldest first, so a leftover that writes again after its channel is released can end on the broken pipe.
