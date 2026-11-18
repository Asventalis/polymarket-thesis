# Data Analysis and Data Mining on Decentralised Prediction Markets

**The Price Process of Polymarket from One Minute to Twelve Hours**

Master's thesis, Master's Degree in Computer Science, Department of Computer Science,
University of Pisa, academic year 2025/2026.
Candidate: Vito Pampinella. Supervisor: Prof. Davide Bacciu.
Co-supervisor: Prof. Giulia Livieri (London School of Economics).

The thesis is [thesis.pdf](thesis.pdf).

The thesis studies the price process of Polymarket, a decentralised prediction market,
over 21 days of order-book and on-chain transaction data (6–26 March 2026). The question
is whether the direction of the next price move, from one minute to twelve hours ahead,
can be predicted from the book, the market's own activity, the transaction graph of
wallets and markets, and sources outside the book. Five families of models are compared
on the same three-class benchmark with a locked test period: causal rules on the price,
stateless learners on hand-made features, a continuous-time stateful model of the book
stream, graph models on the wallet–market transactions, and outside-the-book sources.

Build: `latexmk main.tex` (pdfLaTeX + biber; the output is `build/main.pdf`).
