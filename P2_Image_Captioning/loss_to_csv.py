import re

with open('./training_log.txt', 'r') as t:
    s = t.read()

with open('./data.csv', 'w+') as f:
    for idx, x in enumerate(s.split('\n')):

        epoch = re.findall("(?<=Epoch \[).*(?=\/3\],)", x)[0]
        batch = re.findall("(?<=, Step \[).*(?=\/3236\],)", x)[0]
        loss = re.findall("(?<=Loss: ).*(?=, Perplexity)",x)[0]

        # f.write(epoch + "," + batch + "," + loss + "\n")
        f.write(str(idx + 1) + "," + loss + "\n")