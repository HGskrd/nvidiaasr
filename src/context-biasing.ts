// Contextual biasing for the RNN-T decoder.
//
// We build a prefix trie ("context graph") from the user's bias phrases, after
// tokenizing each phrase into the model's own subword units. During decoding we
// track a single active node per hypothesis and add a log-domain bonus to the
// next token that continues an active partial match. The graph uses failure
// (back-off) arcs so that abandoning a partial match cancels the boost already
// credited to it — that net-zero behaviour is what stops the bias from leaking
// into unrelated words (the failure mode of naive per-token boosting).

export interface ContextNode {
  /** token id -> child node */
  readonly children: Map<number, ContextNode>;
  /** accumulated boost from the root to this node */
  readonly score: number;
  /** true when a bias phrase ends here */
  isEnd: boolean;
}

export class ContextGraph {
  readonly root: ContextNode;
  readonly phraseCount: number;

  private constructor(root: ContextNode, phraseCount: number) {
    this.root = root;
    this.phraseCount = phraseCount;
  }

  /** Build a context graph from token-id sequences, crediting `boost` per token. */
  static build(tokenSequences: number[][], boost: number): ContextGraph {
    const root: ContextNode = { children: new Map(), score: 0, isEnd: false };
    let phraseCount = 0;

    for (const sequence of tokenSequences) {
      if (sequence.length === 0) continue;
      phraseCount++;
      let node = root;
      for (const token of sequence) {
        let next = node.children.get(token);
        if (!next) {
          next = { children: new Map(), score: node.score + boost, isEnd: false };
          node.children.set(token, next);
        }
        node = next;
      }
      node.isEnd = true;
    }

    return new ContextGraph(root, phraseCount);
  }

  get isEmpty(): boolean {
    return this.root.children.size === 0;
  }

  /**
   * Advance the active node by an emitted token and report the log-domain bias
   * for taking it:
   *  - continue a partial match: +(child.score - node.score)
   *  - leave the graph: cancel the banked boost (-node.score), then optionally
   *    restart a fresh match from the root (+rootChild.score).
   * A node that cannot be extended falls back to the root so new matches can
   * start, while keeping the boost already credited.
   */
  step(node: ContextNode, token: number): { node: ContextNode; bias: number } {
    const cont = node.children.get(token);
    if (cont) {
      const next = cont.children.size > 0 ? cont : this.root;
      return { node: next, bias: cont.score - node.score };
    }

    const restart = this.root.children.get(token);
    if (restart) {
      const next = restart.children.size > 0 ? restart : this.root;
      return { node: next, bias: restart.score - node.score };
    }

    return { node: this.root, bias: -node.score };
  }

  /**
   * Per-token additive bonus used to rank expansions from `node`. Equals the
   * matched child's accumulated score (node child overrides a root child); the
   * constant -node.score term from `step` is omitted because it shifts every
   * non-matching token equally and so does not affect ranking among them.
   */
  matchScores(node: ContextNode): Map<number, number> {
    const scores = new Map<number, number>();
    for (const [token, child] of this.root.children) scores.set(token, child.score);
    if (node !== this.root) {
      for (const [token, child] of node.children) scores.set(token, child.score);
    }
    return scores;
  }
}

/** Numerically stable log-softmax over a logits vector. */
export function logSoftmax(logits: Float32Array): Float32Array {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const logNorm = max + Math.log(sum);

  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] - logNorm;
  return out;
}

export interface TokenCandidate {
  token: number;
  logProb: number;
}

/**
 * Top-`k` non-blank tokens ranked by log-prob plus any context match bonus.
 * Returns the raw log-prob (bias is applied separately by the caller so the
 * hypothesis score and the trie advance stay consistent).
 */
export function topTokens(
  logProbs: Float32Array,
  blankId: number,
  k: number,
  matchScores?: Map<number, number>
): TokenCandidate[] {
  const best: Array<{ token: number; rank: number }> = [];
  for (let token = 0; token < logProbs.length; token++) {
    if (token === blankId) continue;
    const rank = logProbs[token] + (matchScores?.get(token) ?? 0);
    if (best.length < k) {
      best.push({ token, rank });
      if (best.length === k) best.sort((a, b) => a.rank - b.rank);
    } else if (rank > best[0].rank) {
      best[0] = { token, rank };
      // keep the smallest rank at index 0
      let i = 0;
      while (i + 1 < best.length && best[i].rank > best[i + 1].rank) {
        const tmp = best[i];
        best[i] = best[i + 1];
        best[i + 1] = tmp;
        i++;
      }
    }
  }
  return best
    .sort((a, b) => b.rank - a.rank)
    .map(({ token }) => ({ token, logProb: logProbs[token] }));
}
