# Automated AI-Based Financial Transaction Categorisation

### Team: Decode

**Team Members:**

* Vinay Pepakayala
* Navaneeth Kola
* Teja Gorrepotu

---

## 📌 Overview

Modern financial applications—from budgeting tools to enterprise accounting—require intelligent classification of raw transaction text such as **“Starbucks,” “Amazon.com,” or “Shell Gas”** into meaningful budget categories like **Food & Beverages, Shopping, or Fuel**.

Most existing solutions rely on costly external APIs, leading to:

* High recurring expenses
* Limited customization
* Latency and reduced control

This project provides a **fully in-house, cost-effective AI system** for fast and accurate transaction classification using **embeddings, fine-tuning, and dynamic similarity-based categorisation.**

---

## 🧠 Problem Statement

A scalable, internal ML solution is required to:

* Eliminate reliance on paid external categorisation APIs
* Enable control over classification logic
* Support evolving custom category taxonomies
* Maintain high accuracy with real-world noisy inputs
* Deliver explainable and auditable categorisation

The goal is to build a **lightweight but business-grade model** for automated financial transaction classification with flexibility, accuracy, and transparency.

---

## 🏗️ Technology Stack

| Component            | Technology                              |
| -------------------- | --------------------------------------- |
| Programming Language | Python                                  |
| ML Model             | `all-mpnet-base-v2` sentence embeddings |
| Frameworks           | PyTorch, Hugging Face Transformers      |
| UI                   | Streamlit                               |
| Storage              | Local JSON/CSV dataset                  |
| Deployment           | Local execution / Extendable API        |

---

## 🧪 System Architecture

This solution uses a **hybrid embedding + similarity classification approach** with optional fine-tuning.

### 1. Text Embeddings

The model converts transaction text into vector representations capturing context and semantic similarity.

Example:

| Raw Transaction Variants | Model Interpretation       |
| ------------------------ | -------------------------- |
| "Uber Ride"              | → Similar embedding        |
| "UBR Trip"               | → Same meaning             |
| "Taxi - Uber"            | → Categorized as Transport |

---

### 2. Embedding-Based Classification

Each label (and its examples) is embedded and similarity-matched to transactions using **cosine similarity**.

Benefits:

* No full model retraining required when categories change
* Dynamic adaptability
* Custom taxonomy support

---

### 3. Fine-Tuning with Synthetic Dataset

To improve real-world robustness, the model was fine-tuned using a custom dataset containing:

* Merchant name variations
* Spelling mistakes and noise
* Short ambiguous descriptions
* Edge-case financial terminology

This improves generalization and reduces false matches.

---

### 4. Evaluation & Optimization

Key enhancements include:

* Similarity thresholding for uncertainty detection
* Confidence scoring
* Review loop for ambiguous classifications
* Reduced confusion between close labels (e.g., Fast Food vs Restaurants)

---

## ✅ Outcome

The final system provides:

* High accuracy across noisy and varied transaction text
* Scalable architecture supporting millions of records via vector indexing (FAISS)
* Continuous category evolution without full retraining
* Offline privacy-first processing

---

## 🔐 Security & Compliance

* Fully offline model → No external API calls
* Optional transaction anonymization
* Suitable for fintech, banking, and personal finance use cases

---

## ⚙️ Scalability & Performance

* Vector search enables fast similarity matching
* Batch processing supported
* Caching avoids repeated embedding generation
* FAISS recommended for large-scale deployment

---

## 🎥 Demo & Source Code

* **GitHub Repository:** *(private repository content placeholder)*

* [https://github.com/vinay-852/GHIC](https://github.com/vinay-852/GHIC)

* **Demo / Prototype Video:**
  [https://drive.google.com/drive/folders/14xRfA45jrdaJMcK7Qr4pw3Iifvwwlii6?usp=sharing](https://drive.google.com/drive/folders/14xRfA45jrdaJMcK7Qr4pw3Iifvwwlii6?usp=sharing)

---

## 🚀 Future Improvements

* Reinforcement learning from user corrections
* Support multilingual merchant text
* Transaction trend forecasting and anomaly detection
* Deployable microservice with REST API support

