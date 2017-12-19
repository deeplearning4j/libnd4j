#Graph

###Basic idea
libnd4j contains graph execution engine, suited for both local and remote execution. However, main mode is execution of externally originated graphs, serialized into FlatBuffers and provided either via pointer, or file.


Basic example
```
auto graph = GraphExecutioner<float>::importFromFlatBuffers("./some_file.fb");
GraphExecutioner<float>::execute(graph);
...
delete graph;
```

###FlatBuffers schemas
You can find scheme files here: https://github.com/deeplearning4j/libnd4j/tree/master/include/graph/scheme
At this moment libnd4j repo contains compiled includes for C++, Python, Java, and JSON, but FlatBuffers can be compiled for PHP, C#, JavaScript, TypeScript and Go as well. Please use `flatc` instructions to do that.



