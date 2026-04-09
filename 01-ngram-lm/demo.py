"""
N-gram Language Model demo on a Shakespeare corpus.

Demonstrates:
  - Training a trigram LM
  - Laplace vs. Kneser-Ney perplexity comparison
  - Text generation with temperature sampling
  - OOV word failure (perplexity spikes to inf or very high)
"""

from ngram_lm import NgramLM

# ---------------------------------------------------------------------------
# Embedded Shakespeare corpus (~400 words from Hamlet and other plays)
# ---------------------------------------------------------------------------

SHAKESPEARE_CORPUS = """
HAMLET: To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of.
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment,
With this regard their currents turn awry,
And lose the name of action. Soft you now,
The fair Ophelia. Nymph, in thy orisons
Be all my sins remembered.
OPHELIA: Good my lord, how does your honour for this many a day?
HAMLET: I humbly thank you; well, well, well.
OPHELIA: My lord, I have remembrances of yours,
That I have longed long to re-deliver;
I pray you, now receive them.
HAMLET: No, not I; I never gave you aught.
OPHELIA: My honour'd lord, you know right well you did;
And, with them, words of so sweet breath composed
As made the things more rich: their perfume lost,
Take these again; for to the noble mind
Rich gifts wax poor when givers prove unkind.
There, my lord.
HAMLET: Ha, ha! are you honest?
OPHELIA: My lord?
HAMLET: Are you fair?
OPHELIA: What means your lordship?
HAMLET: That if you be honest and fair, your honesty should admit no discourse to your beauty.
All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages.
To morrow, and to morrow, and to morrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more: it is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.
Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar.
"""

# Held-out sentence drawn from the same style.
HELD_OUT = "To be or not to be that is the question whether tis nobler in the mind"

# A sentence containing a word that never appeared in training.
OOV_SENTENCE = "zylquorx is a very strange word that the model has never seen before"


def main():
    print("=" * 60)
    print("N-gram Language Model Demo")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. Train
    # -------------------------------------------------------------------
    model = NgramLM()
    model.train(SHAKESPEARE_CORPUS)

    V = model.tokenizer.vocab_size()
    total_unigrams = sum(model.unigram_counts.values())
    total_bigrams = len(model.bigram_counts)
    total_trigrams = len(model.trigram_counts)

    print(f"\nCorpus statistics:")
    print(f"  Vocabulary size : {V}")
    print(f"  Total tokens    : {total_unigrams}")
    print(f"  Unique bigrams  : {total_bigrams}")
    print(f"  Unique trigrams : {total_trigrams}")

    # -------------------------------------------------------------------
    # 2. Perplexity comparison
    # -------------------------------------------------------------------
    print(f"\nHeld-out sentence:")
    print(f"  \"{HELD_OUT}\"")

    ppl_laplace = model.perplexity(HELD_OUT, smoothing="laplace")
    ppl_kn      = model.perplexity(HELD_OUT, smoothing="kneser_ney")

    print(f"\n  Laplace perplexity   : {ppl_laplace:.2f}")
    print(f"  Kneser-Ney perplexity: {ppl_kn:.2f}")
    print()
    if ppl_kn < ppl_laplace:
        print("  -> Kneser-Ney wins (lower is better).")
    else:
        print("  -> Laplace wins on this sentence (unusual; corpus is tiny).")

    # -------------------------------------------------------------------
    # 3. Generate 5 sentences
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generated sentences (prefix='to be', 15 words, temp=0.8)")
    print("=" * 60)

    import numpy as np
    np.random.seed(0)
    for i in range(5):
        sentence = model.generate("to be", n_words=15, smoothing="laplace", temperature=0.8)
        print(f"  [{i+1}] {sentence}")

    # -------------------------------------------------------------------
    # 4. OOV demonstration
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OOV word demonstration")
    print("=" * 60)
    print(f"\nSentence: \"{OOV_SENTENCE}\"")
    print(f"(The word 'zylquorx' was never in the training corpus.)\n")

    ppl_oov_laplace = model.perplexity(OOV_SENTENCE, smoothing="laplace")
    ppl_oov_kn      = model.perplexity(OOV_SENTENCE, smoothing="kneser_ney")

    print(f"  Laplace perplexity   : {ppl_oov_laplace:.2f}")
    print(f"  Kneser-Ney perplexity: {ppl_oov_kn:.2f}")
    print()
    print("  With Laplace, the OOV word still gets probability 1/(C(ctx)+V),")
    print("  so perplexity is very high but finite.")
    print("  With KN, the continuation count for an unseen word is 0,")
    print("  so perplexity spikes to inf (or hits the tiny floor).")
    print()
    if ppl_oov_kn > ppl_oov_laplace or ppl_oov_kn == float("inf"):
        print("  -> Confirmed: OOV sentence is severely penalised.")


if __name__ == "__main__":
    main()
