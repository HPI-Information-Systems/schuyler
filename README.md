# Schuyler: Self-Supervised Clustering of Tables in Relational Databases
Databases are integral to modern applications. They enable the efficient handling and processing of vast amounts of data, making it possible to build services that support millions of users. However, these systems often comprise hundreds of interconnected tables that complicate maintenance and comprehension. To effectively operate and maintain them, having an overview of the database is of utmost importance. Database table clustering involves grouping semantically related tables, which simplifies many database management, analyzing, and integration tasks.

We present Schuyler, a system that clusters database tables by combining structural and semantic features of the database. Specifically, Schuyler finetunes a large language model in a self-supervised manner using triplet-loss to produce high-quality embeddings representing table semantics. Subsequently, these embeddings are clustered to achieve a database table clustering. Our approach requires no labeled training data, and thus is applicable to arbitrary databases.

To validate Schuyler and benchmark it against state-of-the-art competitors, we introduce a benchmark collection consisting of five real-world databases. These databases vary significantly in size (29–481 tables) and complexity (3–47 clusters) and reflect diverse real-world challenges.
Our results demonstrate that Schuyler consistently achieves superior clustering performance, improving the state-of-the-art on average by 0.13 ARI (adjusted Rand index) and 0.10 AMI (adjusted mutual information).

## Data Access
Access data [here](https://my.hidrive.com/share/h88utg.uwd)

## Setup

1. Run ``docker compose up --build``
