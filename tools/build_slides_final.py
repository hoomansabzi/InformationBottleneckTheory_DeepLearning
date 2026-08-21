"""Build slides/slides_final.tex -- the short, self-contained version of the talk.

Takes the full deck (slides/slides.tex) up to and including the "Remove the
randomness entirely" frame, strips every forward reference to material that only
appears in the long version, and appends a results-and-conclusions slide plus a
closing slide.

The full deck is left untouched.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "slides" / "slides.tex"
DST = ROOT / "slides" / "slides_final.tex"

#: Everything up to (not including) this frame is kept.
CUT_AT = r"\begin{frame}{The SNR transition is real --- and irrelevant}"

#: Forward references to material the short deck no longer contains.
STRIP = [
    (
        "{\\footnotesize\\centering\n"
        "A third route --- $k$-nearest-neighbour --- avoids fixing $\\smi^2$ by using only\n"
        "$\\Delta\\I=\\Delta\\HH$, at the price of a different assumption. We will watch that one\n"
        "fail too.\\par}\n",
        "{\\footnotesize\\centering\n"
        "A third route --- $k$-nearest-neighbour --- avoids fixing $\\smi^2$ by using only\n"
        "$\\Delta\\I=\\Delta\\HH$. It pays for that with a different assumption: that the layer has\n"
        "a density at all.\\par}\n",
    ),
    (
        "ReLU (panel B) does not sit at 12: a unit that is off outputs \\textbf{exactly} zero, so\n"
        "some patterns genuinely collide --- a real atom in the distribution, and the reason the\n"
        "$k$-NN estimator breaks later.\\par}",
        "ReLU (panel B) does not sit at 12: a unit that is off outputs \\textbf{exactly} zero, so\n"
        "some patterns genuinely collide --- a real atom in the distribution, and one that breaks\n"
        "any estimator assuming a density.\\par}",
    ),
    (
        "We hit exactly this later: 8 random units at 30 bins ``know'' \\SI{1.0}{bit} about a\n"
        "label they are independent of.",
        "It bites in our own data: 8 random units at 30 bins ``know'' \\SI{1.0}{bit} about a\n"
        "label they are independent of.",
    ),
]

CONCLUSIONS = r"""
% ============================================================== conclusion ==
\section{Conclusions}

\begin{frame}{Results, and what they mean}
\centering\footnotesize
\begin{tabular}{@{}L{3.4cm}L{3.7cm}L{5.1cm}@{}}
\toprule
\textbf{the claim} & \textbf{the test} & \textbf{the result} \\
\midrule
\addlinespace[2pt]
\textbf{1.} Training has two phases, the second being compression &
  change \texttt{tanh}$\to$\texttt{relu}, and nothing else &
  \SI{10.66}{} $\to$ \SI{0.47}{bits}, and the ReLU network is \emph{more} accurate
  (\SI{97.3}{} vs \SI{96.2}{\percent}) \ \no \\
\addlinespace[6pt]
\textbf{2.} Compression causes generalisation &
  exact mutual information in deep linear networks &
  \SI{0.00000}{bits} in both, while $E_g$ differs by $13\times$ (1.61 vs 20.30) \ \no \\
\addlinespace[6pt]
\textbf{3.} SGD's diffusion is the cause &
  deterministic full-batch gradient descent &
  \SI{17.87}{bits} vs \SI{10.66}{} --- \emph{more} compression with no noise at all \ \no \\
\addlinespace[2pt]
\bottomrule
\end{tabular}
\medskip
\begin{columns}[T]
\begin{column}{0.52\textwidth}
\begin{alertblock}{\small The deeper point}
\footnotesize
For a deterministic network $\I(X;T)$ is $\infty$ (continuous $X$) or a \emph{constant}
(discrete $X$) --- measured at machine precision, a flat line at \SI{12}{bits}.\\[3pt]
Every finite number ever plotted is therefore a joint property of the network \textbf{and}
an assumption the analyst added.
\end{alertblock}
\end{column}
\begin{column}{0.46\textwidth}
\begin{block}{\small What survives}
\footnotesize
The saturation is \textbf{real}: \texttt{tanh} units genuinely do pile up against
$\pm 1$ (\SI{0}{} $\to$ \SI{75.4}{\percent}).\\[3pt]
What fails is the step from that to \emph{``the layer discarded ten bits about the
input''}.
\end{block}
\end{column}
\end{columns}
\end{frame}

\begin{frame}{One slide to take away}
\begin{beamercolorbox}[sep=8pt,center,rounded=true,shadow=true]{block title}
\large
Hold the network fixed. Change only how you \emph{measure} it.\\[4pt]
\normalsize
bin edges: \ layer 5 compresses \SI{5.59}{bits} \ $\longrightarrow$ \ \SI{0.32}{bits}\\
bin count: \ the same layer, \SI{0.10}{} \ $\longrightarrow$ \ \SI{2.27}{bits}, non-monotone\\
full precision: \ a flat line at \SI{12}{bits}, no dynamics at all\\
rescale the weights, same function: \ \SI{4.32}{} \ $\longrightarrow$ \ \SI{0.03}{bits}
\end{beamercolorbox}
\bigskip
\begin{center}
\large
The information plane is a picture of a network \emph{and} a ruler.\\[3pt]
\normalsize
Before attributing a trajectory to learning, check it is not a property of the ruler.
\end{center}
\end{frame}

\begin{frame}[plain]
\begin{center}
\Huge Thank you\\[10pt]
\normalsize Questions?
\end{center}
\end{frame}

\end{document}
"""


def main() -> None:
    src = SRC.read_text()

    idx = src.index(CUT_AT)
    kept = src[:idx]

    # drop the trailing "\section{Claim 3...}" comment banner if it now heads nothing
    kept = kept.rstrip() + "\n"

    kept = kept.replace(
        "\\usepackage{booktabs}",
        "\\usepackage{booktabs}\n\\usepackage{array}\n"
        "\\newcolumntype{L}[1]{>{\\raggedright\\arraybackslash}p{#1}}",
        1,
    )

    for old, new in STRIP:
        if old not in kept:
            raise SystemExit(f"forward reference not found, refusing to guess:\n{old[:80]}")
        kept = kept.replace(old, new, 1)

    kept = kept.replace(
        r"\subtitle{A reproduction and critical study}",
        r"\subtitle{A reproduction and critical study}",
    )

    DST.write_text(kept + CONCLUSIONS)
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
