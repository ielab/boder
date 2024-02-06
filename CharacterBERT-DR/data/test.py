with open('queries/docleaderboard-queries.tsv', 'r') as f1, open('dl2019/queries.dl2019.tsv', 'r') as f2:
    c = 0
    query_set = set()
    for line in f1:
        qid, query = line.strip().split('\t')
        query_set.add(query)

    for line in f2:
        qid, query = line.strip().split('\t')
        if query in query_set:
            c+=1
    print(c)


