use anyhow::Result;
use clap::Parser;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::thread_local;
use vibrato::{Dictionary, Tokenizer};
use zstd::stream::read::Decoder;

/// Command line arguments
#[derive(Parser, Debug)]
struct Args {
    /// Input format: "plain" for plain text or "jsonl" for JSON Lines
    #[arg(short, long, default_value = "plain")]
    format: String,

    /// Minimum frequency to display in the histogram
    #[arg(short, long, default_value_t = 1)]
    min_freq: u32,

    /// Number of tokens from end of sentence to include in the match not including search tokens
    #[arg(short, long)]
    num_tokens: Option<usize>,

    /// Filter by genre (can be specified multiple times)
    #[arg(short, long)]
    genre: Vec<String>,

    /// Output delimiter: "\t" for tab, ": " for colon-space, or any other string
    #[arg(short, long, default_value = "\\t")]
    output_delimiter: String,

    /// Token join string
    #[arg(short, long, default_value = "_")]
    join_string: String,

    /// Ignore punctuation symbols
    #[arg(long)]
    ignore_punctuation: bool,

    /// Path to the dictionary file or directory containing it
    #[arg(long)]
    dict_path: String,

    /// Number of top results to display
    #[arg(short, long)]
    top_k: Option<usize>,

    /// Number of threads to use for parallel processing
    #[arg(long, default_value_t = 1)]
    num_threads: usize,

    /// Output the genre as the first column
    #[arg(long)]
    output_genre: bool,

    /// Tokens to be searched for
    #[arg(long, default_value = "もの,こと,と")]
    search_tokens: String,

    /// Output mode: "default" or "csv"
    #[arg(long, default_value = "default")]
    output_mode: String,

    /// Maximum number of tokens to include in a match
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,

    /// Enable n-gram extraction mode
    #[arg(long)]
    compute_ngrams: bool,
}

// Global dictionary data
static DICT_DATA: Lazy<Vec<u8>> = Lazy::new(|| {
    let dict_path = std::env::var("DICT_PATH").unwrap_or_else(|_| Args::parse().dict_path);
    load_dictionary_data(&dict_path)
});

thread_local! {
    static TOKENIZER: Tokenizer = {
        let dict = Dictionary::read(BufReader::new(DICT_DATA.as_slice())).expect("Failed to read dictionary");
        Tokenizer::new(dict).ignore_space(true).expect("Failed to create tokenizer")
    };
}

fn load_dictionary_data(dict_path: &str) -> Vec<u8> {
    if dict_path.ends_with(".zst") {
        let reader = File::open(dict_path).expect("Failed to open dictionary file");
        let mut decoder = Decoder::new(reader).expect("Failed to create decoder");
        let mut dict_data = Vec::new();
        io::copy(&mut decoder, &mut dict_data).expect("Failed to copy dictionary data");
        dict_data
    } else if dict_path.ends_with(".dic") {
        fs::read(dict_path).expect("Failed to read dictionary file")
    } else if Path::new(dict_path).is_dir() {
        let zst_path = Path::new(dict_path).join("system.dic.zst");
        let dic_path = Path::new(dict_path).join("system.dic");
        if zst_path.exists() {
            let reader = File::open(zst_path).expect("Failed to open dictionary file");
            let mut decoder = Decoder::new(reader).expect("Failed to create decoder");
            let mut dict_data = Vec::new();
            io::copy(&mut decoder, &mut dict_data).expect("Failed to copy dictionary data");
            dict_data
        } else if dic_path.exists() {
            fs::read(dic_path).expect("Failed to read dictionary file")
        } else {
            panic!("No valid dictionary file found in the directory");
        }
    } else {
        panic!("Invalid dictionary path");
    }
}

fn parse_escape_sequences(s: &str) -> String {
    s.replace("\\t", "\t").replace("\\n", "\n")
}

fn read_lines_from_stdin<R: BufRead>(
    reader: R,
    format: &str,
    genre: &[String],
) -> impl ParallelIterator<Item = String> + Send {
    let lines = reader.lines();

    let iter = if format == "jsonl" {
        lines
            .filter_map(move |line| {
                if let Ok(line) = line {
                    let value: Value = serde_json::from_str(&line).ok()?;
                    let genres = value.get("genre")?.as_array()?;
                    let genres: Vec<String> = genres
                        .iter()
                        .filter_map(|g| g.as_str().map(|s| s.to_string()))
                        .collect();

                    if genre.is_empty() || genre.iter().any(|g| genres.contains(g)) {
                        return Some(
                            value
                                .get("sentences")?
                                .as_array()?
                                .iter()
                                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                                .collect::<Vec<String>>(),
                        );
                    }
                }
                None
            })
            .flat_map(|vec| vec.into_iter())
            .collect::<Vec<_>>()
            .into_par_iter()
    } else {
        lines
            .filter_map(|line| line.ok().map(|s| vec![s]))
            .flat_map(|vec| vec.into_iter())
            .collect::<Vec<_>>()
            .into_par_iter()
    };

    iter
}

fn process_sentences<I>(
    sentences: I,
    args: &Args,
    join_string: &str,
    search_tokens: &HashSet<String>,
) -> HashMap<String, u32>
where
    I: ParallelIterator<Item = String> + Send,
{
    sentences
        .into_par_iter()
        .map(|sentence| {
            TOKENIZER.with(|tokenizer| {
                let mut worker = tokenizer.new_worker();
                worker.reset_sentence(&sentence);
                worker.tokenize();

                let num_tokens = worker.num_tokens();
                if num_tokens == 0 {
                    return HashMap::new();
                }

                let mut end_index = num_tokens;

                if args.ignore_punctuation {
                    while end_index > 0 {
                        let feature = worker.token(end_index - 1).feature();
                        if feature.split(',').next() == Some("補助記号") {
                            end_index -= 1;
                        } else {
                            break;
                        }
                    }
                }

                let mut local_histogram = HashMap::new();
                let mut stack = Vec::new();

                for i in (0..end_index).rev() {
                    let token = worker.token(i);
                    let surface = token.surface().to_string();
                    let feature = token.feature();
                    if feature.split(',').next() == Some("補助記号") {
                        continue;
                    }

                    if surface == "、" || surface == "，" || surface == "," {
                        break;
                    }

                    stack.push(surface.clone());

                    if search_tokens.contains(&surface) {
                        let pattern = stack
                            .iter()
                            .rev()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(&join_string);
                        *local_histogram.entry(pattern).or_insert(0) += 1;
                        break;
                    }

                    if let Some(max_tokens) = args.num_tokens {
                        if stack.len() > max_tokens {
                            break;
                        }
                    }
                }

                local_histogram
            })
        })
        .reduce(HashMap::new, |mut acc, local_histogram| {
            for (pattern, count) in local_histogram {
                *acc.entry(pattern).or_insert(0) += count;
            }
            acc
        })
}

fn extract_ngrams_from_histogram(
    histogram: HashMap<String, u32>,
    max_tokens: usize,
    join_string: &str,
) -> HashMap<String, u32> {
    histogram
        .par_iter()
        .map(|(pattern, &count)| {
            let tokens: Vec<&str> = pattern.split(join_string).collect();
            let mut ngram_counts = HashMap::new();

            for i in 0..tokens.len() {
                for j in i..(i + max_tokens).min(tokens.len()) {
                    let ngram_len = j - i + 1;
                    if ngram_len > max_tokens {
                        break;
                    }
                    let ngram: Vec<&str> = tokens[i..=j].to_vec();
                    let ngram_string = ngram.join(join_string);
                    *ngram_counts.entry(ngram_string).or_insert(0) += count;
                }
            }

            ngram_counts
        })
        .reduce(HashMap::new, |mut acc, ngram_counts| {
            for (ngram, count) in ngram_counts {
                *acc.entry(ngram).or_insert(0) += count;
            }
            acc
        })
}

fn filter_and_sort_histogram(histogram: HashMap<String, u32>, min_freq: u32) -> Vec<(String, u32)> {
    let filtered_histogram: Vec<_> = histogram
        .into_par_iter()
        .filter(|&(_, count)| count >= min_freq)
        .collect();

    let mut sorted_histogram = filtered_histogram;
    sorted_histogram.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
    sorted_histogram
}

fn output_histogram_results(
    filtered_histogram: Vec<(String, u32)>,
    delimiter: &str,
    top_k: Option<usize>,
    genres: &str,
    output_mode: &str,
    max_columns: usize,
) {
    let mut header = Vec::new();
    let mut results_to_output = Vec::new();
    let mut current_count = 0;

    if let Some(top_k) = top_k {
        for (pattern, count) in filtered_histogram.iter() {
            if results_to_output.len() >= top_k && *count < current_count {
                break;
            }
            results_to_output.push((pattern.clone(), *count));
            current_count = *count;
        }
    } else {
        results_to_output = filtered_histogram
            .iter()
            .map(|(pattern, count)| (pattern.clone(), *count))
            .collect();
    }

    if output_mode == "csv" {
        header.clear();
        if !genres.is_empty() {
            header.push("genre".to_string());
        }
        header.push("count".to_string());
        for i in 0..max_columns {
            header.push(format!("t{}", i));
        }
        println!("{}", header.join(delimiter));
    }

    for (pattern, count) in results_to_output {
        let tokens: Vec<&str> = pattern.split(&delimiter).collect();
        let mut row = Vec::new();
        if !genres.is_empty() {
            row.push(genres.to_string());
        }
        row.push(count.to_string());
        row.extend(tokens.iter().cloned().map(|s| s.to_string()));
        while row.len() < max_columns + 2 {
            row.push("".to_string());
        }
        println!("{}", row.join(delimiter));
    }
}

fn run(args: Args, reader: impl BufRead) -> Result<()> {
    let delimiter = parse_escape_sequences(&args.output_delimiter);
    let join_string = parse_escape_sequences(&args.join_string);
    let genres = args.genre.join(",");
    let search_tokens: HashSet<String> = args.search_tokens.split(',').map(String::from).collect();

    // Configure the number of threads for Rayon
    ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .unwrap();

    let sentences = read_lines_from_stdin(reader, &args.format, &args.genre);
    // let sentences_progress = sentences.tqdm();

    let histogram = process_sentences(sentences, &args, &join_string, &search_tokens);
    let filtered_histogram = filter_and_sort_histogram(histogram.clone(), args.min_freq);

    if args.compute_ngrams {
        let ngram_counts = extract_ngrams_from_histogram(histogram, args.max_tokens, &join_string);
        let filtered_ngram_counts = filter_and_sort_histogram(ngram_counts, args.min_freq);
        let max_columns = filtered_ngram_counts
            .iter()
            .map(|(pattern, _)| pattern.split(&delimiter).count())
            .max()
            .unwrap_or(0);
        output_histogram_results(
            filtered_ngram_counts,
            &delimiter,
            args.top_k,
            &genres,
            &args.output_mode,
            max_columns,
        );
    } else {
        let max_columns = filtered_histogram
            .iter()
            .map(|(pattern, _)| pattern.split(&delimiter).count())
            .max()
            .unwrap_or(0);
        output_histogram_results(
            filtered_histogram,
            &delimiter,
            args.top_k,
            &genres,
            &args.output_mode,
            max_columns,
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args, io::stdin().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init() {
        INIT.call_once(|| {
            // Set the environment variable for the dictionary path
            std::env::set_var(
                "DICT_PATH",
                std::env::var("DICT_PATH").unwrap_or_else(|_| Args::parse().dict_path),
            );
            // Initialize the dictionary data
            let _ = Lazy::force(&DICT_DATA);
        });
    }

    #[test]
    fn test_parse_escape_sequences() {
        assert_eq!(parse_escape_sequences("\\t"), "\t");
        assert_eq!(parse_escape_sequences("\\n"), "\n");
        assert_eq!(parse_escape_sequences(" "), " ");
    }

    #[test]
    fn test_read_lines_from_stdin() {
        init();
        let data = "hello\nworld\n";
        let cursor = io::Cursor::new(data);
        let result = read_lines_from_stdin(cursor, "plain", &[]).collect::<Vec<_>>();
        assert_eq!(result, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_process_sentences() {
        init();
        let args = Args {
            format: "plain".to_string(),
            min_freq: 1,
            num_tokens: None,
            genre: vec![],
            output_delimiter: "\\t".to_string(),
            join_string: "_".to_string(),
            ignore_punctuation: false,
            dict_path: "test.dic".to_string(),
            top_k: None,
            num_threads: 1,
            output_genre: false,
            search_tokens: "もの,こと,と".to_string(),
            output_mode: "default".to_string(),
            max_tokens: 4,
            compute_ngrams: false,
        };
        let search_tokens: HashSet<String> =
            args.search_tokens.split(',').map(String::from).collect();
        let sentences = vec!["ものがいえる".to_string(), "ことができる".to_string()];
        let result = process_sentences(sentences.into_par_iter(), &args, "_", &search_tokens);
        println!("{:?}", result);
        assert!(result.contains_key("もの_が_いえる"));
        assert!(result.contains_key("こと_が_できる"));
    }

    #[test]
    fn test_extract_ngrams_from_histogram() {
        init();
        let histogram = vec![
            ("もの_が_いえる".to_string(), 1),
            ("こと_が_できる".to_string(), 1),
        ]
        .into_iter()
        .collect();
        let result = extract_ngrams_from_histogram(histogram, 4, "_");
        println!("{:?}", result);
        assert_eq!(result.get("もの").cloned().unwrap_or(0), 1);
        assert_eq!(result.get("こと_が_できる").cloned().unwrap_or(0), 1);
    }

    #[test]
    fn test_filter_and_sort_histogram() {
        let histogram = vec![("もの".to_string(), 2), ("こと".to_string(), 1)]
            .into_iter()
            .collect();
        let result = filter_and_sort_histogram(histogram, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "もの");
        assert_eq!(result[1].0, "こと");
    }
}
