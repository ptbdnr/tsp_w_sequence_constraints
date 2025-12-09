# Vehicle Routing Problem

## Travelling Salesman Problem with Sequence Constraints and Non-Linear Objective Function

### Problem Definition

Given a set of nodes $𝑁 = {0,1,2, … , 𝑛, 𝑛 + 1}$ and the distances between each pair of nodes $𝑑_{𝑖𝑗}$, $𝑖, 𝑗 ∈ 𝑁$, then a feasible route should:

* start form node 0
* visit each node exactly once
* return to node $𝑛 + 1$ at the end of the route

and:
* traveling form node $𝑖 ∈ 𝑁  \ {0, 𝑛 + 1}$ to node $𝑗 ∈ 𝑁 \ {0, 𝑛 + 1}$ is forbidden when: (i) $𝑖$ is an even number, (ii) $𝑗$ is an odd number, and (iii) $𝑖 < 𝑛/2$, and
* traveling form node $𝑖 ∈ 𝑁 \ {0, 𝑛 + 1}$ to node $𝑗 ∈ 𝑁 \ {0, 𝑛 + 1}$ is forbidden when: (i) $𝑖$ is an odd number, (ii) $𝑗$ is an even number, and (iii) $𝑖 ≥ 𝑛/2$

Find the route that minimises the function: $𝐿 ∙ 𝛥 + 𝐷$, where 

(i) $𝐿 = max_{𝑖,𝑗∈𝑁}{𝑑_{𝑖𝑗}} ∙ 𝑛$,

(ii) $𝐷$ is the total distance of the route, and

(iii) $𝛥$ is the difference between:

* (a) $𝑚𝑎𝑥𝐷$ = the longest node pair distance and,
* (b) $𝑚𝑖𝑛𝐷$ = the shortest node pair distance of a given route.

For instance, if a route consists of node pairs: (0,3), (3,1), (1,2), (2,4), (4,5) with respective node pair distances: $𝑑_{03} = 10$, $𝑑_{31} = 5$, $𝑑_{12} = 9$, $𝑑_{24} = 7$, $𝑑_{45} = 8$, then $𝛥 = 𝑑_{03} − 𝑑_{31} = 5$ and the total distance is equal to 39. Note that in the latter example $𝑛 = 4$.

### Instructions

* Design and code a program that can read a CSV file of the form: `Node_ID,x-cor,y-cor`

* Calculate the Euclidean distances between nodes (rounded up to the first decimal place)

* Make it clear in your answers if you have made any assumptions about the data/problem

* Provide a solution to the above problem, i.e. Find a feasible route that minimises the aforementioned function, in a txt file using the following format: 
```text
Route:0-3-1-2
Total Distance: 849.25
Delta Value: 12.38
```

* Write a short description of your method and respective results

* Follow-up questions

    * (Q1) How is your method / program scaling?
    * (Q2) What is the required time by your method/ program to get a good solution?
    * (Q3) How would you change your method / program if you were asked to significantly reduce the computational time?
    * (Q4) How would you change your method / program to deal with dynamic arrival of requests?
    * (Q5) What aspects of the problem could be enhanced to include stochastic information?

### Assumptions

* node IDs are integers
* list of node IDs always start with "0" (text) or 0 (numeric)
* there is always a feasible solution (to validate)
* undirected distances, the distance between any two nodes is symetric, i.e. $d_{ij} = d_{ji}$
* the basic problem can be deployed on arbitrary hardware, i.e. no constraints on memory and compute.

### Observations

* complete distance matrix requires $n²/2$ memory, where $n+1$ is the number of nodes
* not all nodes can connect to the terminating $n+1$
* $𝐿$ is static in the basic problem, but may change if a node is added (subject to the existing and new coordinates)
* the problem is reducable to the well-known TSP (by simplifying the objective function and removing constraints), TSP has $O(n!)$ feasible solutions, and known to be NP-hard, i.e., in the worst case the solution requires $O(n!)$ time. To solve real-size problems under time-constraint, we have to trade accuracy (guarantee for optimal solution) for speed.

### Backlog

* Construction Heuristic: Enumerate to find optimal solution
* Metaheuristics: GA/SA/TS/ACO/VNS
* Monitoring/Logging: callback, checkpoints, temination criteria
* Operators: Two/Three Opt Swap without sequence reverse, path relinking
* consider a cluster-first-then-route approach: cluster the nodes based on proximity to convert the distance matrix to a sparse matrix
* distributed population evaluation: distribute computation on multiple compute resources, for example, move distance calculations to GPU cluster
* identifying dominance based on distance can be accelerated: instead of computing euclidean (4x read operations, 3x multiplications, 1x addition), it can be terminated early if partial computation (2x read op + 1x multiplication) proves sufficient data for dominance

### Out of scope

* the objective function should be re-considered if the stochastic information refers to existence of a node or its coordinate
