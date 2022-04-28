# Appendix


## More experiments


### Span Enumeration Length Limitation

We evaluate different span enumeration length limits on ACE04 datasets. With small span enumeration length limitations, the span-based predictor cannot propose long entities, and it is difficult for multi-scale features to model the deep hierarchical structure of natural language sentences. However, enumerating long spans is computationally expensive and may introduce noise in cross-attention. We find that $L=16$ is an optimal choice.

| Length | Loc. F1 | Cls. F1 |   Pr  |  Rec. |   F1  |
|--------|:-------:|:-------:|:-----:|:-----:|:-----:|
| 8      |  91.74  |  90.92  | 86.97 | 87.28 | 87.12 |
| 12     |  90.93  |  90.71  | 87.83 | 86.82 | 87.32 |
| 16     |  **92.34**  |  **91.75**  | **87.90** | **88.34** | **88.12** |
| 20     |  92.16  |  91.30  | 87.41 | 87.81 | 87.61 |
| 24     |  91.90  |  91.23  | 87.35 | 87.61 | 87.48 |

<!---
\begin{table}[htb]
\small
\centering
\begin{tabular}{lccccc}
\toprule
Length & Loc. F1 & Cls. F1 & Pr & Rec. & F1  \\
\midrule
8     & 91.74 & 90.92 & 86.97 & 87.28 & 87.12 \\
12    & 90.93 & 90.71 & 87.83 & 86.82 & 87.32 \\
% 14    & 91.00 & 89.83 & 86.53 & 85.73 & 86.13 \\
16    & \textbf{92.34} & \textbf{91.75} & \textbf{87.90} & \textbf{88.34} & \textbf{88.12} \\
% 18    & 90.95 & 89.72 & 86.11 & 85.57 & 85.84 \\
20    & 92.16 & 91.30 & 87.41 & 87.81 & 87.61 \\
24    & 91.90 & 91.23 & 87.35 & 87.61 & 87.48 \\
\bottomrule
\end{tabular}
\caption{Effect of the different span enumeration length limits. Evaluate on ACE04 dataset. }
\label{tab:FpnLayerACE04}
\end{table}
--->

### Number of Layers of the Transformer Decoder

We try different numbers of layers of the transformer decoder on the ACE04 dataset. The transformer decoder needs multiple layers to refine the coarse proposals iteratively. We find that $M=3$ is an optimal choice.

| Layers |  Loc. F1  |  Cls. F1  |     Pr    |    Rec.   |     F1    |
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|
| 1      |   92.05   |   91.11   |   86.88   |   87.74   |   87.31   |
| 2      |   92.01   |   91.26   | **88.01** |   87.51   |   87.76   |
| 3      | **92.34** | **91.75** |   87.90   | **88.34** | **88.12** |
| 4      |   91.63   |   90.73   |   87.23   |   86.89   |   87.06   |
| 5      |   91.80   |   90.55   |   87.16   |   86.79   |   86.97   |

<!---
\begin{table}[htb]
\small
\centering
\begin{tabular}{lccccc}
\toprule
Layers & Loc. F1 & Cls. F1 & Pr & Rec. & F1  \\
\midrule
1    & 92.05 & 91.11 & 86.88 & 87.74 & 87.31 \\
2    & 92.01 & 91.26 & \textbf{88.01} & 87.51 & 87.76 \\
3    & \textbf{92.34} & \textbf{91.75} & 87.90 & \textbf{88.34} & \textbf{88.12} \\
4    & 91.63 & 90.73 & 87.23 & 86.89 & 87.06 \\
5    & 91.80 & 90.55 & 87.16 & 86.79 & 86.97 \\
\bottomrule
\end{tabular}
\caption{Effect of the different number of layers of the transformer decoder. Evaluate on ACE04 dataset. }
\label{tab:SSNLayerACE04}
\end{table}
--->

### More Multi-Scale Cross-Attention Weight

We show more examples from the ACE04 dataset to illustrate the multi-scale attention weight of PnRNet. We visualize the cross-attention weight of a certain attention head in the last decoder layer.

In Figure 1, the span with the highest score is "Lemieux" (the predicted entity of the proposal). Another two high-score phrases are "Jordan" (the person highly related to "Lemieux") and "part-owner of a team" (an entity related to "law professor").

In Figure 2, three spans with the highest scores are "law professor rick pildes" (the predicted entity of the proposal), "law professor", "rick pildes" (two sub-constituents of the entity "law professor rick pildes").

Through these cases, we can see that multi-scale features can model the hierarchical structure of natural language sentences well and can help a lot in predicting nested named entities. In our PnRNet, multi-scale features are used to provide richer hierarchical contextual information for locally contextualized entity proposals and achieve better performance compared with token-level features.

![Image1](ACE04S314Q3H2-attn.png "Fig1. The query of the illustrated cross-attention weight is an entity proposal that finally predicts 'Lemieux'(PER).")

![Image2](ACE04S496Q7H7-attn.png "Fig2. The query of the illustrated cross-attention weight is an entity proposal that finally predicts 'law professor rick pildes'(PER).")

<!---
\begin{figure}[htb]
  \centering
  \includegraphics[width=0.65\linewidth]{ACE04S314Q3H2-attn.png}
  \caption{The query of the illustrated cross-attention weight is an entity proposal that finally predicts ``Lemieux"(PER).}
  \label{fig:MoreVis1}
\end{figure}

\begin{figure}[htb]
  \centering
  \includegraphics[width=0.65\linewidth]{ACE04S496Q7H7-attn.png}
  \caption{The query of the illustrated cross-attention weight is an entity proposal that finally predicts ``law professor rick pildes"(PER).}
  \label{fig:MoreVis2}
\end{figure}
--->

## Dataset Statistics

The following table shows the statistics of the datasets we used in experiments. We show (1) the number of sentences, (2) the average sentence length, (3) the ratio of nested entities, (4) the average number of entities in a sentence, (5) the maximum number of entities in a sentence, (6) average entity length, and (7) maximum entity length.

|               | ace04-train | ace04-dev | ace04-test | ace05-train | ace05-dev | ace05-test | genia-train | genia-test | kbp17-train | kbp17-dev | kbp17-test | connl03-train | connl03-dev | connl03-test |
|---------------|-------------|-----------|------------|-------------|-----------|------------|-------------|------------|-------------|-----------|------------|---------------|-------------|--------------|
| #sent         | 6200        | 745       | 812        | 7194        | 969       | 1047       | 16692       | 1854       | 10546       | 545       | 4267       | 10773         | 3250        | 3453         |
| avg sent len  | 22.5        | 23.0      | 23.1       | 19.2        | 18.9      | 17.2       | 25.3        | 26.0       | 19.6        | 20.6      | 19.3       | 15.9          | 15.8        | 13.4         |
| nested ratio  | 45.7        | 43.4      | 46.7       | 38.4        | 34.8      | 37.4       | 17.9        | 21.8       | 28.1        | 32.2      | 29.4       | -             | -           | -            |
| avg #e / sent | 3.6         | 3.4       | 3.7        | 3.4         | 3.3       | 2.9        | 3.0         | 3.0        | 3.0         | 3.4       | 3.0        | 2.1           | 1.8         | 1.6          |
| max #e / sent | 28          | 22        | 20         | 27          | 23        | 17         | 25          | 14         | 58          | 15        | 21         | 20            | 20          | 31           |
| avg e-len     | 2.6         | 2.7       | 2.7        | 2.3         | 2.1       | 2.3        | 2.0         | 2.1        | 1.9         | 2.1       | 2.0        | 1.4           | 1.4         | 1.4          |
| max e-len     | 63          | 40        | 44         | 49          | 31        | 27         | 18          | 15         | 45          | 32        | 49         | 10            | 10          | 6            |

<!---
\begin{table*}[htb]
\centering
\small
\begin{tabular}{lm{0.7cm}m{0.6cm}m{0.6cm}m{0.7cm}m{0.6cm}m{0.7cm}m{0.8cm}m{0.7cm}m{0.8cm}m{0.6cm}m{0.7cm}m{0.8cm}m{0.7cm}m{0.7cm}}
\toprule \multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Statistics\end{tabular}} & \multicolumn{3}{c}{ACE04}                                                          &  \multicolumn{3}{c}{ACE05}                                                          & \multicolumn{2}{c}{GENIA}                                                          & \multicolumn{3}{c}{KBP17}                                                          &
\multicolumn{3}{c}{CoNLL03}                                                           \\
\cmidrule(lr){2-4}  \cmidrule(lr){5-7}   \cmidrule(lr){8-9}   \cmidrule(lr){10-12}   \cmidrule(lr){13-15}
& train & dev & test & train & dev & test & train & test & train & dev & test & train & dev & test \\
\midrule
\#sent         & 6200 & 745  & 812  & 7194 & 969  & 1047 & 16692 & 1854 & 10546 & 545  & 4267 & 10773 & 3250 & 3453  \\
avg sent len   & 22.5 & 23.0 & 23.1 & 19.2 & 18.9 & 17.2 & 25.3  & 26.0 & 19.6  & 20.6 & 19.3 & 15.9  & 15.8 & 13.4  \\
nested ratio   & 45.7 & 43.4 & 46.7 & 38.4 & 34.8 & 37.4 & 17.9  & 21.8 & 28.1  & 32.2 & 29.4 & -     & -    & -     \\
avg \#e / sent & 3.6  & 3.4  & 3.7  & 3.4  & 3.3  & 2.9  & 3.0   & 3.0  & 3.0   & 3.4  & 3.0  & 2.1   & 1.8  & 1.6   \\
max \#e / sent & 28   & 22   & 20   & 27   & 23   & 17   & 25    & 14   & 58    & 15   & 21   & 20    & 20   & 31    \\
avg e-len      & 2.6  & 2.7  & 2.7  & 2.3  & 2.1  & 2.3  & 2.0   & 2.1  & 1.9   & 2.1  & 2.0  & 1.4   & 1.4  & 1.4   \\
max e-len      & 63   & 40   & 44   & 49   & 31   & 27   & 18    & 15   & 45    & 32   & 49   & 10    & 10   & 6    \\
\bottomrule
\end{tabular}
\caption{Statistics of datasets we used in experiments.}
\label{tab:DsetStat}
\end{table*}
--->
