Note - there is a clear optimization where we can use the cgraphs generated in the first pass... This would enable the use of "action.reinforce" and would only work if gradient is done on the same process as inference.

Right now we toss it and do an extra evaluation call.