## Annotation Guidlines

1. **Name**: The name of the dataset.
2. **Link**: direct link to the dataset. If the dataset is hosted in HuggingFace, we can use the same link in both the Link and HF Link fields. If there is a link from GitHub and a link from HuggingFace, put the link from GitHub. 
3. **HF_Link**: The huggingface link of the dataset.
4. **License**: the license of the dataset. 
5. **Year**: the year the paper was published.
6. **Language**: is the dataset English or multilingual.
7. **Domain**: is the source or the content  of the data; for example, Domain=Wikipedia means the dataset is extracted from Wikipedia and news articles, if the dataset is extracted from news outlets
8. **Form**: can be text, spoken, images, or videos.
9. **Collection_Style**: is how the dataset is extracted, and the annotation strategy of the dataset.
10. **Description**: a short description of the dataset. 
11. **Volume**: The volume is the total number of samples in the dataset. In the case the dataset is multilingual it equals the number of samples in specified language if it exists. 
12. **Unit**: We use 'Units' = sentences if the dataset has short samples, even if there are multiple sentences. Unit=documents is usually for datasets that have documents, like language modelling, topic classification, etc. For datasets that have multiple inputs, like question answering, which has (question, context), then we usually use the question to indicate the Unit and Volume i.e., Unit=sentences, Volume=number of questions.. 
13. **Ethical_Risks**: "Low": "most likely no ethical risks associated with this dataset", "Medium": "social media datasets or web-extracted datasets", "High": "hate/offensive datasets from social media, or web pages".
14. **Provider**: The provider is the main entity that contributed to the dataset. If all the authors have one affiliation, then that affiliation is the Provider, for example, Google, Stanford, etc. If there are multiple affiliations, then if the host of the dataset is for a given entity like Google/dataset_name, then we use Google as the Provider. For any other examples, we list all the affiliations. We can also use the funds in the acknowledgment section to determine the Provider. 
15. **Derived_From**: lists all the datasets that were used to create or derive the current dataset.
16. **Paper_Title**: The title of the paper.
17. **Paper_Link**: we use the PDF link of the dataset, for example, https://arxiv.org/pdf/2504.21677
18. **Tokenized**: Is the dataset lemmatized or stemmed?
19. **Host**: you can use the Link attribute to write the repository where the dataset is hosted. Use other if the host is not from the given options. 
20. **Access**: Free if the dataset is free. Upon Request, if the dataset is behind a form. With-Fee if the dataset is paid. 
21. **Cost**: the cost of the dataset if paid. 
22. **Test_Split**: is true only if the dataset has (training and test splits). If the dataset has only one split, even if it is for testing, we set that to false
23. **Tasks**: the list of tasks that this dataset is intended for. Use other if the task doesn’t exist in the options.
24. **Venue_Title**: full name of the venue, for arXiv we don't use the full name.
25. **Venue_Type**: type, either preprint, workshop, conference, or journal.
26. **Venue_Name**: if the dataset is from arXiv and there is no mention of the conference workshop, etc. in the paper. If the paper has something like (paper published in ACL), then we put Venue Title = ACL, Venue Type = conference, and Venue Name = Association of Computational Linguistics.
27. **Authors**: the authors of the paper in an ordered fashion.
28. **Affiliations**: list only the affiliations without repetition
29. **Abstract**: the full abstract of the paper. 