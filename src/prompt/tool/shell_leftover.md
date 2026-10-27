## Leftover processes

A command that exits while a process it started still holds its output channel is reported FINISHED — its own exit status and the output above are the result. The leftover is killed and no further output from it is collected. {{mode_sentence}}

---

[leftover process] the command exited; its own exit status and the output above are its result. A process it left behind still holds its output channel, so no more of that output can be collected. The leftover is left running — mahbot holds that channel open so a closed pipe cannot end it, keeping the 16 most recent such channels and releasing the oldest first, so a leftover that writes again after its channel is released can end on the broken pipe.
