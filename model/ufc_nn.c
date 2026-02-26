/*
 * UFC Fight Winner Prediction Neural Network
 * 
 * Predicts the probability that fighter1 wins based on statistical deltas
 * between fighter attributes (height, reach, striking, takedown stats, etc.)
 * 
 * Architecture:
 *   Input: 20 features (fighter stat deltas + matchup context features)
 *   Hidden: 64 -> 32 neurons (tanh activation)
 *   Output: 1 neuron (sigmoid) : P(fighter1 wins)
 * 
 * Dataset: Trained on 30 years of UFC fight history (1994-2023)
 * Source: ../data/ufc_complete_dataset.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>

#define FIGHTER_STATS_DICT_PATH "../data/fighter_stats_dict.json"

/* Network architecture */
#define INPUT_SIZE    20  /* fighter stat deltas + matchup context features */
#define HIDDEN_1_SIZE 64
#define HIDDEN_2_SIZE 32
#define OUTPUT_SIZE   1

/* Feature indices for clarity */
#define FEAT_HEIGHT_DELTA       0
#define FEAT_REACH_DELTA        1
#define FEAT_AGE_DELTA          2
#define FEAT_SIG_STRIKE_PM_DELTA 3
#define FEAT_SIG_STRIKE_ACC_DELTA 4
#define FEAT_SIG_STRIKE_ABS_DELTA 5
#define FEAT_SIG_STRIKE_DEF_DELTA 6
#define FEAT_TAKEDOWN_AVG_DELTA  7
#define FEAT_TAKEDOWN_ACC_DELTA  8
#define FEAT_TAKEDOWN_DEF_DELTA  9
#define FEAT_SUB_AVG_DELTA      10
#define FEAT_WEIGHT_DELTA       11
#define FEAT_STRIKING_ADVANTAGE 12  /* derived: strike output - strike absorbed */
#define FEAT_GRAPPLING_SCORE    13  /* derived: takedown + submission combination */
#define FEAT_WIN_RATE_DELTA     14  /* prior career win-rate difference */
#define FEAT_TOTAL_WINS_DELTA   15  /* prior career total wins difference */
#define FEAT_TOTAL_FIGHTS_DELTA 16  /* prior career fight-count difference */
#define FEAT_WEIGHTED_SCORE_DELTA 17 /* win-rate weighted by experience */
#define FEAT_SUB_STYLE_DELTA    18  /* submission-specialist style bias */
#define FEAT_H2H_DELTA          19  /* direct head-to-head prior bias */

typedef struct FighterRecord {
    char name[64];
    char weight_class[64];
    int wins;
    int losses;
    int total;
} FighterRecord;

typedef struct HeadToHeadRecord {
    char name_a[64];
    char name_b[64];
    char weight_class[64];
    int wins_a;
    int wins_b;
    int total;
} HeadToHeadRecord;

typedef struct MatchContext {
    FighterRecord *fighters;
    int fighter_count;
    int fighter_cap;
    HeadToHeadRecord *h2h;
    int h2h_count;
    int h2h_cap;
} MatchContext;

typedef struct UFCFight {
    char event_date[32];
    char weight_class[64];
    char fighter1[64];
    char fighter2[64];
    char outcome[16];  /* "fighter1", "fighter2", "Draw" */
    
    /* Fighter 1 stats */
    double f1_height;
    double f1_reach;
    double f1_age;
    double f1_sig_strikes_pm;
    double f1_sig_strikes_acc;
    double f1_sig_strikes_abs;
    double f1_sig_strikes_def;
    double f1_takedown_avg;
    double f1_takedown_acc;
    double f1_takedown_def;
    double f1_sub_avg;
    double f1_weight;
    
    /* Fighter 2 stats */
    double f2_height;
    double f2_reach;
    double f2_age;
    double f2_sig_strikes_pm;
    double f2_sig_strikes_acc;
    double f2_sig_strikes_abs;
    double f2_sig_strikes_def;
    double f2_takedown_avg;
    double f2_takedown_acc;
    double f2_takedown_def;
    double f2_sub_avg;
    double f2_weight;
    
    int label;  /* 1 if fighter1 won, 0 if fighter2 won, -1 for draw/nocontest */
} UFCFight;

typedef struct ClassStats {
    char weight_class[64];
    int total;
    int correct;
} ClassStats;

typedef struct WeightClassBucket {
    char weight_class[64];
    UFCFight *fights;
    int count;
    int cap;
} WeightClassBucket;

static int is_allowed_weight_class(const char *weight_class) {
    return weight_class && weight_class[0] != '\0';
}

typedef struct Model {
    long double w1[INPUT_SIZE][HIDDEN_1_SIZE], b1[HIDDEN_1_SIZE];
    long double w2[HIDDEN_1_SIZE][HIDDEN_2_SIZE], b2[HIDDEN_2_SIZE];
    long double w3[HIDDEN_2_SIZE][OUTPUT_SIZE], b3[OUTPUT_SIZE];
    
    /* Momentum velocities */
    long double v_w1[INPUT_SIZE][HIDDEN_1_SIZE], v_b1[HIDDEN_1_SIZE];
    long double v_w2[HIDDEN_1_SIZE][HIDDEN_2_SIZE], v_b2[HIDDEN_2_SIZE];
    long double v_w3[HIDDEN_2_SIZE][OUTPUT_SIZE], v_b3[OUTPUT_SIZE];
    
    /* Activations */
    long double h1[HIDDEN_1_SIZE], h2[HIDDEN_2_SIZE], output[OUTPUT_SIZE];
    
    /* Feature normalization statistics (computed from training data) */
    double feat_mean[INPUT_SIZE];
    double feat_std[INPUT_SIZE];
    
    int num_trained_samples;
} Model;

int save_model(Model *m, const char *path);
int load_model(Model *m, const char *path);

volatile sig_atomic_t keep_running = 1;
void handle_sigint(int sig) { (void)sig; keep_running = 0; }

/* Activation functions */
long double relu(long double x) { return x > 0.0L ? x : 0.0L; }
long double relu_deriv(long double x) { return x > 0.0L ? 1.0L : 0.0L; }
long double tanh_act(long double x) { return tanhl(x); }
long double tanh_deriv(long double y) { return 1.0L - y * y; }
long double sigmoid(long double x) { return 1.0L / (1.0L + expl(-x)); }
long double sigmoid_deriv(long double y) { return y * (1.0L - y); }

static long double temperature_scale_probability(long double prob, long double temperature) {
    const long double eps = 1e-12L;
    if (temperature <= 1.0L) return prob;
    if (prob < eps) prob = eps;
    if (prob > 1.0L - eps) prob = 1.0L - eps;
    long double logit = logl(prob / (1.0L - prob));
    return sigmoid(logit / temperature);
}

/* He initialization for ReLU/tanh networks */
long double he_init(int fan_in) {
    return ((long double)rand() / RAND_MAX * 2.0L - 1.0L) * sqrtl(2.0L / (long double)fan_in);
}

/* CSV parsing helper - extract field at column index */
static int parse_csv_field(const char *line, int col_idx, char *out, int out_size) {
    int current_col = 0;
    int i = 0;
    int out_idx = 0;
    int in_quotes = 0;
    
    while (line[i] && current_col < col_idx) {
        if (line[i] == '"') {
            in_quotes = !in_quotes;
        } else if (line[i] == ',' && !in_quotes) {
            current_col++;
        }
        i++;
    }
    
    if (current_col != col_idx) return -1;
    
    /* Extract the field */
    if (line[i] == '"') {
        i++;
        in_quotes = 1;
    }
    
    while (line[i] && out_idx < out_size - 1) {
        if (in_quotes && line[i] == '"' && line[i+1] == '"') {
            out[out_idx++] = '"';
            i += 2;
        } else if (in_quotes && line[i] == '"') {
            break;
        } else if (!in_quotes && line[i] == ',') {
            break;
        } else {
            out[out_idx++] = line[i++];
        }
    }
    out[out_idx] = '\0';
    return 0;
}

static int compare_fight_date(const void *a, const void *b) {
    const UFCFight *fa = (const UFCFight *)a;
    const UFCFight *fb = (const UFCFight *)b;
    return strcmp(fa->event_date, fb->event_date);
}

static int name_equals_ci(const char *a, const char *b) {
    if (!a || !b) {
        return 0;
    }
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static int month_to_number(const char *month) {
    static const char *months[] = {
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    };
    for (int i = 0; i < 12; i++) {
        if (name_equals_ci(month, months[i])) {
            return i + 1;
        }
    }
    return 0;
}

static int event_date_to_key(const char *event_date) {
    if (!event_date || !event_date[0]) {
        return 0;
    }

    int year = 0, month = 0, day = 0;
    if (sscanf(event_date, "%d-%d-%d", &year, &month, &day) == 3) {
        return year * 10000 + month * 100 + day;
    }

    char month_name[32] = {0};
    if (sscanf(event_date, "%31[^ ] %d, %d", month_name, &day, &year) == 3) {
        month = month_to_number(month_name);
        if (month > 0) {
            return year * 10000 + month * 100 + day;
        }
    }
    return 0;
}

static int infer_latest_weight_class(UFCFight *fights,
                                     int num_fights,
                                     const char *fighter1,
                                     const char *fighter2,
                                     char *out_class,
                                     size_t out_size) {
    int latest_f1 = 0;
    int latest_f2 = 0;
    char class_f1[64] = {0};
    char class_f2[64] = {0};

    for (int i = 0; i < num_fights; i++) {
        int key = event_date_to_key(fights[i].event_date);
        if (key <= 0) {
            continue;
        }

        if (name_equals_ci(fights[i].fighter1, fighter1) || name_equals_ci(fights[i].fighter2, fighter1)) {
            if (key > latest_f1) {
                latest_f1 = key;
                strncpy(class_f1, fights[i].weight_class, sizeof(class_f1) - 1);
            }
        }

        if (name_equals_ci(fights[i].fighter1, fighter2) || name_equals_ci(fights[i].fighter2, fighter2)) {
            if (key > latest_f2) {
                latest_f2 = key;
                strncpy(class_f2, fights[i].weight_class, sizeof(class_f2) - 1);
            }
        }
    }

    if (latest_f1 == 0 && latest_f2 == 0) {
        return 0;
    }

    if (latest_f1 > 0 && latest_f2 > 0) {
        if (strcmp(class_f1, class_f2) == 0) {
            strncpy(out_class, class_f1, out_size - 1);
            return 1;
        }
        if (latest_f1 >= latest_f2) {
            strncpy(out_class, class_f1, out_size - 1);
        } else {
            strncpy(out_class, class_f2, out_size - 1);
        }
        return 2;
    }

    if (latest_f1 > 0) {
        strncpy(out_class, class_f1, out_size - 1);
    } else {
        strncpy(out_class, class_f2, out_size - 1);
    }
    return 1;
}

static int fighter_exists_in_history(UFCFight *fights, int num_fights, const char *fighter_name) {
    if (!fighter_name || fighter_name[0] == '\0') {
        return 0;
    }
    for (int i = 0; i < num_fights; i++) {
        if (name_equals_ci(fights[i].fighter1, fighter_name) || name_equals_ci(fights[i].fighter2, fighter_name)) {
            return 1;
        }
    }
    return 0;
}

static int lookup_fighter_stats_from_dict(const char *dict_path,
                                          const char *weight_class,
                                          const char *fighter_name,
                                          double stats_out[12]) {
    FILE *f = fopen(dict_path, "r");
    if (!f) return 0;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return 0;
    }
    rewind(f);

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return 0;
    }
    size_t read_count = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[read_count] = '\0';

    char key[256];
    snprintf(key, sizeof(key), "\"%s|%s\":{", weight_class, fighter_name);
    char *entry = strstr(buf, key);
    if (!entry) {
        free(buf);
        return 0;
    }

    char *obj = strstr(entry, "{\"height\":");
    if (!obj) {
        free(buf);
        return 0;
    }

    int parsed = sscanf(
        obj,
        "{\"height\":%lf,\"reach\":%lf,\"age\":%lf,\"sig_str_pm\":%lf,\"sig_acc\":%lf,\"sig_abs\":%lf,\"sig_def\":%lf,\"td_avg\":%lf,\"td_acc\":%lf,\"td_def\":%lf,\"sub_avg\":%lf,\"weight\":%lf}",
        &stats_out[0], &stats_out[1], &stats_out[2], &stats_out[3],
        &stats_out[4], &stats_out[5], &stats_out[6], &stats_out[7],
        &stats_out[8], &stats_out[9], &stats_out[10], &stats_out[11]);

    free(buf);
    return parsed == 12;
}

static void apply_fighter_stats(UFCFight *fight, int fighter_index, const double stats[12]) {
    if (fighter_index == 1) {
        fight->f1_height = stats[0];
        fight->f1_reach = stats[1];
        fight->f1_age = stats[2];
        fight->f1_sig_strikes_pm = stats[3];
        fight->f1_sig_strikes_acc = stats[4];
        fight->f1_sig_strikes_abs = stats[5];
        fight->f1_sig_strikes_def = stats[6];
        fight->f1_takedown_avg = stats[7];
        fight->f1_takedown_acc = stats[8];
        fight->f1_takedown_def = stats[9];
        fight->f1_sub_avg = stats[10];
        fight->f1_weight = stats[11];
    } else {
        fight->f2_height = stats[0];
        fight->f2_reach = stats[1];
        fight->f2_age = stats[2];
        fight->f2_sig_strikes_pm = stats[3];
        fight->f2_sig_strikes_acc = stats[4];
        fight->f2_sig_strikes_abs = stats[5];
        fight->f2_sig_strikes_def = stats[6];
        fight->f2_takedown_avg = stats[7];
        fight->f2_takedown_acc = stats[8];
        fight->f2_takedown_def = stats[9];
        fight->f2_sub_avg = stats[10];
        fight->f2_weight = stats[11];
    }
}

static int prompt_fighter_stats(const char *label, UFCFight *fight, int fighter_index, char *buffer, size_t buffer_size) {
    printf("%s stats (height reach age sig_str_pm sig_acc sig_abs sig_def td_avg td_acc td_def sub_avg weight)\n> ", label);
    fflush(stdout);
    if (fgets(buffer, (int)buffer_size, stdin) == NULL) return 0;

    double vals[12];
    if (sscanf(buffer, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], &vals[5],
               &vals[6], &vals[7], &vals[8], &vals[9], &vals[10], &vals[11]) != 12) {
        return -1;
    }

    apply_fighter_stats(fight, fighter_index, vals);
    return 1;
}

static void init_context(MatchContext *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}

static void free_context(MatchContext *ctx) {
    free(ctx->fighters);
    ctx->fighters = NULL;
    ctx->fighter_count = 0;
    ctx->fighter_cap = 0;
    free(ctx->h2h);
    ctx->h2h = NULL;
    ctx->h2h_count = 0;
    ctx->h2h_cap = 0;
}

static int ensure_fighter_cap(MatchContext *ctx) {
    if (ctx->fighter_count < ctx->fighter_cap) {
        return 0;
    }
    int next_cap = ctx->fighter_cap == 0 ? 128 : ctx->fighter_cap * 2;
    FighterRecord *next = realloc(ctx->fighters, (size_t)next_cap * sizeof(FighterRecord));
    if (!next) {
        return -1;
    }
    ctx->fighters = next;
    ctx->fighter_cap = next_cap;
    return 0;
}

static int ensure_h2h_cap(MatchContext *ctx) {
    if (ctx->h2h_count < ctx->h2h_cap) {
        return 0;
    }
    int next_cap = ctx->h2h_cap == 0 ? 128 : ctx->h2h_cap * 2;
    HeadToHeadRecord *next = realloc(ctx->h2h, (size_t)next_cap * sizeof(HeadToHeadRecord));
    if (!next) {
        return -1;
    }
    ctx->h2h = next;
    ctx->h2h_cap = next_cap;
    return 0;
}

static FighterRecord *get_or_add_fighter(MatchContext *ctx, const char *name, const char *weight_class) {
    for (int i = 0; i < ctx->fighter_count; i++) {
        if (strcmp(ctx->fighters[i].name, name) == 0 && strcmp(ctx->fighters[i].weight_class, weight_class) == 0) {
            return &ctx->fighters[i];
        }
    }
    if (ensure_fighter_cap(ctx) != 0) {
        return NULL;
    }
    FighterRecord *record = &ctx->fighters[ctx->fighter_count++];
    memset(record, 0, sizeof(*record));
    strncpy(record->name, name, sizeof(record->name) - 1);
    strncpy(record->weight_class, weight_class, sizeof(record->weight_class) - 1);
    return record;
}

static const FighterRecord *find_fighter_record(const MatchContext *ctx, const char *name, const char *weight_class) {
    for (int i = 0; i < ctx->fighter_count; i++) {
        if (strcmp(ctx->fighters[i].name, name) == 0 && strcmp(ctx->fighters[i].weight_class, weight_class) == 0) {
            return &ctx->fighters[i];
        }
    }
    return NULL;
}

static HeadToHeadRecord *get_or_add_h2h(MatchContext *ctx, const char *fighter1, const char *fighter2, const char *weight_class) {
    const char *left = fighter1;
    const char *right = fighter2;
    if (strcmp(left, right) > 0) {
        left = fighter2;
        right = fighter1;
    }

    for (int i = 0; i < ctx->h2h_count; i++) {
        if (strcmp(ctx->h2h[i].name_a, left) == 0 &&
            strcmp(ctx->h2h[i].name_b, right) == 0 &&
            strcmp(ctx->h2h[i].weight_class, weight_class) == 0) {
            return &ctx->h2h[i];
        }
    }

    if (ensure_h2h_cap(ctx) != 0) {
        return NULL;
    }
    HeadToHeadRecord *record = &ctx->h2h[ctx->h2h_count++];
    memset(record, 0, sizeof(*record));
    strncpy(record->name_a, left, sizeof(record->name_a) - 1);
    strncpy(record->name_b, right, sizeof(record->name_b) - 1);
    strncpy(record->weight_class, weight_class, sizeof(record->weight_class) - 1);
    return record;
}

static void copy_fighter_stats(UFCFight *dst, int dst_side, const UFCFight *src, int src_side) {
    double *d_height = (dst_side == 1) ? &dst->f1_height : &dst->f2_height;
    double *d_reach = (dst_side == 1) ? &dst->f1_reach : &dst->f2_reach;
    double *d_age = (dst_side == 1) ? &dst->f1_age : &dst->f2_age;
    double *d_sig_pm = (dst_side == 1) ? &dst->f1_sig_strikes_pm : &dst->f2_sig_strikes_pm;
    double *d_sig_acc = (dst_side == 1) ? &dst->f1_sig_strikes_acc : &dst->f2_sig_strikes_acc;
    double *d_sig_abs = (dst_side == 1) ? &dst->f1_sig_strikes_abs : &dst->f2_sig_strikes_abs;
    double *d_sig_def = (dst_side == 1) ? &dst->f1_sig_strikes_def : &dst->f2_sig_strikes_def;
    double *d_td_avg = (dst_side == 1) ? &dst->f1_takedown_avg : &dst->f2_takedown_avg;
    double *d_td_acc = (dst_side == 1) ? &dst->f1_takedown_acc : &dst->f2_takedown_acc;
    double *d_td_def = (dst_side == 1) ? &dst->f1_takedown_def : &dst->f2_takedown_def;
    double *d_sub = (dst_side == 1) ? &dst->f1_sub_avg : &dst->f2_sub_avg;
    double *d_weight = (dst_side == 1) ? &dst->f1_weight : &dst->f2_weight;

    const double s_height = (src_side == 1) ? src->f1_height : src->f2_height;
    const double s_reach = (src_side == 1) ? src->f1_reach : src->f2_reach;
    const double s_age = (src_side == 1) ? src->f1_age : src->f2_age;
    const double s_sig_pm = (src_side == 1) ? src->f1_sig_strikes_pm : src->f2_sig_strikes_pm;
    const double s_sig_acc = (src_side == 1) ? src->f1_sig_strikes_acc : src->f2_sig_strikes_acc;
    const double s_sig_abs = (src_side == 1) ? src->f1_sig_strikes_abs : src->f2_sig_strikes_abs;
    const double s_sig_def = (src_side == 1) ? src->f1_sig_strikes_def : src->f2_sig_strikes_def;
    const double s_td_avg = (src_side == 1) ? src->f1_takedown_avg : src->f2_takedown_avg;
    const double s_td_acc = (src_side == 1) ? src->f1_takedown_acc : src->f2_takedown_acc;
    const double s_td_def = (src_side == 1) ? src->f1_takedown_def : src->f2_takedown_def;
    const double s_sub = (src_side == 1) ? src->f1_sub_avg : src->f2_sub_avg;
    const double s_weight = (src_side == 1) ? src->f1_weight : src->f2_weight;

    *d_height = s_height;
    *d_reach = s_reach;
    *d_age = s_age;
    *d_sig_pm = s_sig_pm;
    *d_sig_acc = s_sig_acc;
    *d_sig_abs = s_sig_abs;
    *d_sig_def = s_sig_def;
    *d_td_avg = s_td_avg;
    *d_td_acc = s_td_acc;
    *d_td_def = s_td_def;
    *d_sub = s_sub;
    *d_weight = s_weight;
}

static int load_latest_fighter_snapshot(UFCFight *fights, int num_fights, const char *weight_class, const char *fighter_name, UFCFight *query, int query_side) {
    int best_idx = -1;
    int best_side = 0;

    for (int i = 0; i < num_fights; i++) {
        if (strcmp(fights[i].weight_class, weight_class) != 0) {
            continue;
        }

        int hit_side = 0;
        if (strcmp(fights[i].fighter1, fighter_name) == 0) {
            hit_side = 1;
        } else if (strcmp(fights[i].fighter2, fighter_name) == 0) {
            hit_side = 2;
        }
        if (hit_side == 0) {
            continue;
        }

        if (best_idx < 0 || strcmp(fights[i].event_date, fights[best_idx].event_date) > 0) {
            best_idx = i;
            best_side = hit_side;
        }
    }

    if (best_idx < 0) {
        return -1;
    }

    copy_fighter_stats(query, query_side, &fights[best_idx], best_side);
    return 0;
}

static void print_side_by_side_stats(const UFCFight *fight) {
    printf("\n%-22s | %-18s | %-18s\n", "Metric", fight->fighter1, fight->fighter2);
    printf("%-22s-+-%-18s-+-%-18s\n", "----------------------", "------------------", "------------------");
    printf("%-22s | %18.6f | %18.6f\n", "Height", fight->f1_height, fight->f2_height);
    printf("%-22s | %18.6f | %18.6f\n", "Reach", fight->f1_reach, fight->f2_reach);
    printf("%-22s | %18.6f | %18.6f\n", "Age", fight->f1_age, fight->f2_age);
    printf("%-22s | %18.6f | %18.6f\n", "Sig strikes / min", fight->f1_sig_strikes_pm, fight->f2_sig_strikes_pm);
    printf("%-22s | %18.6f | %18.6f\n", "Sig strike accuracy", fight->f1_sig_strikes_acc, fight->f2_sig_strikes_acc);
    printf("%-22s | %18.6f | %18.6f\n", "Sig absorbed / min", fight->f1_sig_strikes_abs, fight->f2_sig_strikes_abs);
    printf("%-22s | %18.6f | %18.6f\n", "Sig defense", fight->f1_sig_strikes_def, fight->f2_sig_strikes_def);
    printf("%-22s | %18.6f | %18.6f\n", "Takedown avg", fight->f1_takedown_avg, fight->f2_takedown_avg);
    printf("%-22s | %18.6f | %18.6f\n", "Takedown accuracy", fight->f1_takedown_acc, fight->f2_takedown_acc);
    printf("%-22s | %18.6f | %18.6f\n", "Takedown defense", fight->f1_takedown_def, fight->f2_takedown_def);
    printf("%-22s | %18.6f | %18.6f\n", "Submission avg", fight->f1_sub_avg, fight->f2_sub_avg);
    printf("%-22s | %18.6f | %18.6f\n", "Weight", fight->f1_weight, fight->f2_weight);
}

static int fighter_stats_identical(const UFCFight *fight, double eps) {
    return fabs(fight->f1_height - fight->f2_height) <= eps &&
           fabs(fight->f1_reach - fight->f2_reach) <= eps &&
           fabs(fight->f1_age - fight->f2_age) <= eps &&
           fabs(fight->f1_sig_strikes_pm - fight->f2_sig_strikes_pm) <= eps &&
           fabs(fight->f1_sig_strikes_acc - fight->f2_sig_strikes_acc) <= eps &&
           fabs(fight->f1_sig_strikes_abs - fight->f2_sig_strikes_abs) <= eps &&
           fabs(fight->f1_sig_strikes_def - fight->f2_sig_strikes_def) <= eps &&
           fabs(fight->f1_takedown_avg - fight->f2_takedown_avg) <= eps &&
           fabs(fight->f1_takedown_acc - fight->f2_takedown_acc) <= eps &&
           fabs(fight->f1_takedown_def - fight->f2_takedown_def) <= eps &&
           fabs(fight->f1_sub_avg - fight->f2_sub_avg) <= eps &&
           fabs(fight->f1_weight - fight->f2_weight) <= eps;
}

static double prior_win_rate(const FighterRecord *record) {
    if (!record || record->total == 0) {
        return 0.5;
    }
    return (record->wins + 1.0) / (record->total + 2.0);
}

static double weighted_score(const FighterRecord *record) {
    if (!record || record->total == 0) {
        return 0.0;
    }
    return prior_win_rate(record) * log1p((double)record->total);
}

static double head_to_head_bias(const HeadToHeadRecord *record, const char *fighter1, const char *fighter2) {
    if (!record || record->total == 0) {
        return 0.0;
    }

    int f1_wins = 0;
    int f2_wins = 0;
    if (strcmp(record->name_a, fighter1) == 0 && strcmp(record->name_b, fighter2) == 0) {
        f1_wins = record->wins_a;
        f2_wins = record->wins_b;
    } else if (strcmp(record->name_a, fighter2) == 0 && strcmp(record->name_b, fighter1) == 0) {
        f1_wins = record->wins_b;
        f2_wins = record->wins_a;
    }

    return (double)(f1_wins - f2_wins) / (double)record->total;
}

static void update_context_with_result(MatchContext *ctx, UFCFight *fight) {
    FighterRecord *r1 = get_or_add_fighter(ctx, fight->fighter1, fight->weight_class);
    FighterRecord *r2 = get_or_add_fighter(ctx, fight->fighter2, fight->weight_class);
    HeadToHeadRecord *h = get_or_add_h2h(ctx, fight->fighter1, fight->fighter2, fight->weight_class);
    if (!r1 || !r2 || !h) {
        return;
    }

    r1->total++;
    r2->total++;
    h->total++;

    if (fight->label == 1) {
        r1->wins++;
        r2->losses++;
        if (strcmp(h->name_a, fight->fighter1) == 0) {
            h->wins_a++;
        } else {
            h->wins_b++;
        }
    } else if (fight->label == 0) {
        r2->wins++;
        r1->losses++;
        if (strcmp(h->name_a, fight->fighter2) == 0) {
            h->wins_a++;
        } else {
            h->wins_b++;
        }
    }
}

static ClassStats *get_or_add_class_stats(ClassStats **stats, int *count, int *cap, const char *weight_class) {
    for (int i = 0; i < *count; i++) {
        if (strcmp((*stats)[i].weight_class, weight_class) == 0) {
            return &(*stats)[i];
        }
    }

    if (*count >= *cap) {
        int next_cap = (*cap == 0) ? 16 : (*cap * 2);
        ClassStats *next = realloc(*stats, (size_t)next_cap * sizeof(ClassStats));
        if (!next) {
            return NULL;
        }
        *stats = next;
        *cap = next_cap;
    }

    ClassStats *slot = &(*stats)[(*count)++];
    memset(slot, 0, sizeof(*slot));
    if (weight_class && weight_class[0] != '\0') {
        strncpy(slot->weight_class, weight_class, sizeof(slot->weight_class) - 1);
    } else {
        strncpy(slot->weight_class, "Unknown", sizeof(slot->weight_class) - 1);
    }
    return slot;
}

static int sanitize_class_name(const char *src, char *dst, size_t dst_size) {
    if (!src || dst_size < 2) {
        return -1;
    }
    size_t out = 0;
    for (size_t i = 0; src[i] != '\0' && out + 1 < dst_size; i++) {
        unsigned char ch = (unsigned char)src[i];
        if (isalnum(ch)) {
            dst[out++] = (char)tolower(ch);
        } else {
            dst[out++] = '_';
        }
    }
    if (out == 0) {
        dst[out++] = 'u';
    }
    dst[out] = '\0';
    return 0;
}

static int build_class_model_path(const char *weight_class, char *path, size_t path_size) {
    char slug[128];
    if (sanitize_class_name(weight_class, slug, sizeof(slug)) != 0) {
        return -1;
    }
    int written = snprintf(path, path_size, "ufc_model_%s.bin", slug);
    return (written > 0 && (size_t)written < path_size) ? 0 : -1;
}

static WeightClassBucket *get_or_add_bucket(WeightClassBucket **buckets, int *count, int *cap, const char *weight_class) {
    for (int i = 0; i < *count; i++) {
        if (strcmp((*buckets)[i].weight_class, weight_class) == 0) {
            return &(*buckets)[i];
        }
    }

    if (*count >= *cap) {
        int next_cap = (*cap == 0) ? 16 : (*cap * 2);
        WeightClassBucket *next = realloc(*buckets, (size_t)next_cap * sizeof(WeightClassBucket));
        if (!next) {
            return NULL;
        }
        *buckets = next;
        *cap = next_cap;
    }

    WeightClassBucket *bucket = &(*buckets)[(*count)++];
    memset(bucket, 0, sizeof(*bucket));
    strncpy(bucket->weight_class, weight_class, sizeof(bucket->weight_class) - 1);
    return bucket;
}

static int bucket_push_fight(WeightClassBucket *bucket, UFCFight *fight) {
    if (bucket->count >= bucket->cap) {
        int next_cap = (bucket->cap == 0) ? 128 : (bucket->cap * 2);
        UFCFight *next = realloc(bucket->fights, (size_t)next_cap * sizeof(UFCFight));
        if (!next) {
            return -1;
        }
        bucket->fights = next;
        bucket->cap = next_cap;
    }
    bucket->fights[bucket->count++] = *fight;
    return 0;
}

static void free_buckets(WeightClassBucket *buckets, int count) {
    if (!buckets) {
        return;
    }
    for (int i = 0; i < count; i++) {
        free(buckets[i].fights);
        buckets[i].fights = NULL;
        buckets[i].count = 0;
        buckets[i].cap = 0;
    }
    free(buckets);
}

/* Load UFC dataset from CSV */
static int load_ufc_data(const char *path, UFCFight **fights, int *count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return -1;
    }
    
    char line[32768];
    int cap = 1000;
    *fights = malloc(cap * sizeof(UFCFight));
    *count = 0;
    
    /* Skip header */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }
    
    char field[256];
    while (fgets(line, sizeof(line), f)) {
        if (*count >= cap) {
            cap *= 2;
            *fights = realloc(*fights, cap * sizeof(UFCFight));
        }
        
        UFCFight *fight = &(*fights)[*count];
        memset(fight, 0, sizeof(UFCFight));
        
          /* Parse CSV columns based on ufc_fights_full_with_odds.csv structure */
          /* Key columns: event_date(1), weight_class(2), outcome(3), fighter_a_name(4), fighter_b_name(5),
              fighter_a_age(6), fighter_a_height(7), fighter_a_reach(8), fighter_a_weight(9),
              fighter_b_age(11), fighter_b_height(12), fighter_b_reach(13), fighter_b_weight(14),
              fighter_a_sig_strikes_landed(50), fighter_a_sig_strikes_attempted(51),
              fighter_a_takedowns_landed(52), fighter_a_takedowns_attempted(53),
              fighter_a_submission_attempts(56), fighter_a_fight_minutes(57),
              fighter_b_sig_strikes_landed(58), fighter_b_sig_strikes_attempted(59),
              fighter_b_takedowns_landed(60), fighter_b_takedowns_attempted(61),
              fighter_b_submission_attempts(64), fighter_b_fight_minutes(65) */

          parse_csv_field(line, 1, fight->event_date, sizeof(fight->event_date));
          parse_csv_field(line, 2, fight->weight_class, sizeof(fight->weight_class));
          parse_csv_field(line, 4, fight->fighter1, sizeof(fight->fighter1));
          parse_csv_field(line, 5, fight->fighter2, sizeof(fight->fighter2));
          parse_csv_field(line, 3, fight->outcome, sizeof(fight->outcome));

          parse_csv_field(line, 6, field, sizeof(field));
          fight->f1_age = atof(field);
          parse_csv_field(line, 7, field, sizeof(field));
          fight->f1_height = atof(field);
          parse_csv_field(line, 8, field, sizeof(field));
          fight->f1_reach = atof(field);
          parse_csv_field(line, 9, field, sizeof(field));
          fight->f1_weight = atof(field);

          parse_csv_field(line, 11, field, sizeof(field));
          fight->f2_age = atof(field);
          parse_csv_field(line, 12, field, sizeof(field));
          fight->f2_height = atof(field);
          parse_csv_field(line, 13, field, sizeof(field));
          fight->f2_reach = atof(field);
          parse_csv_field(line, 14, field, sizeof(field));
          fight->f2_weight = atof(field);

          double f1_sig_landed, f1_sig_attempted, f1_td_landed, f1_td_attempted, f1_sub_attempts, f1_minutes;
          double f2_sig_landed, f2_sig_attempted, f2_td_landed, f2_td_attempted, f2_sub_attempts, f2_minutes;

          parse_csv_field(line, 50, field, sizeof(field));
          f1_sig_landed = atof(field);
          parse_csv_field(line, 51, field, sizeof(field));
          f1_sig_attempted = atof(field);
          parse_csv_field(line, 52, field, sizeof(field));
          f1_td_landed = atof(field);
          parse_csv_field(line, 53, field, sizeof(field));
          f1_td_attempted = atof(field);
          parse_csv_field(line, 56, field, sizeof(field));
          f1_sub_attempts = atof(field);
          parse_csv_field(line, 57, field, sizeof(field));
          f1_minutes = atof(field);

          parse_csv_field(line, 58, field, sizeof(field));
          f2_sig_landed = atof(field);
          parse_csv_field(line, 59, field, sizeof(field));
          f2_sig_attempted = atof(field);
          parse_csv_field(line, 60, field, sizeof(field));
          f2_td_landed = atof(field);
          parse_csv_field(line, 61, field, sizeof(field));
          f2_td_attempted = atof(field);
          parse_csv_field(line, 64, field, sizeof(field));
          f2_sub_attempts = atof(field);
          parse_csv_field(line, 65, field, sizeof(field));
          f2_minutes = atof(field);

          if (f1_minutes <= 0.0 || f2_minutes <= 0.0) {
                continue;
          }

          fight->f1_sig_strikes_pm = f1_sig_landed / f1_minutes;
          fight->f2_sig_strikes_pm = f2_sig_landed / f2_minutes;
          fight->f1_sig_strikes_acc = (f1_sig_attempted > 0.0) ? (f1_sig_landed / f1_sig_attempted) : 0.0;
          fight->f2_sig_strikes_acc = (f2_sig_attempted > 0.0) ? (f2_sig_landed / f2_sig_attempted) : 0.0;
          fight->f1_sig_strikes_abs = f2_sig_landed / f1_minutes;
          fight->f2_sig_strikes_abs = f1_sig_landed / f2_minutes;
          fight->f1_sig_strikes_def = (f2_sig_attempted > 0.0) ? (1.0 - (f2_sig_landed / f2_sig_attempted)) : 0.0;
          fight->f2_sig_strikes_def = (f1_sig_attempted > 0.0) ? (1.0 - (f1_sig_landed / f1_sig_attempted)) : 0.0;
          fight->f1_takedown_avg = (f1_td_landed / f1_minutes) * 15.0;
          fight->f2_takedown_avg = (f2_td_landed / f2_minutes) * 15.0;
          fight->f1_takedown_acc = (f1_td_attempted > 0.0) ? (f1_td_landed / f1_td_attempted) : 0.0;
          fight->f2_takedown_acc = (f2_td_attempted > 0.0) ? (f2_td_landed / f2_td_attempted) : 0.0;
          fight->f1_takedown_def = (f2_td_attempted > 0.0) ? (1.0 - (f2_td_landed / f2_td_attempted)) : 0.0;
          fight->f2_takedown_def = (f1_td_attempted > 0.0) ? (1.0 - (f1_td_landed / f1_td_attempted)) : 0.0;
          fight->f1_sub_avg = (f1_sub_attempts / f1_minutes) * 15.0;
          fight->f2_sub_avg = (f2_sub_attempts / f2_minutes) * 15.0;
        
        /* Determine label */
        if (strcmp(fight->outcome, "1") == 0 || strstr(fight->outcome, "fighter1")) {
            fight->label = 1;
        } else if (strcmp(fight->outcome, "0") == 0 || strstr(fight->outcome, "fighter2")) {
            fight->label = 0;
        } else {
            fight->label = -1;  /* Draw or no contest */
        }
        
        /* Filter out fights with missing critical data or draws */
        if (fight->label >= 0 && fight->f1_height > 0 && fight->f2_height > 0 &&
            fight->f1_age > 0 && fight->f2_age > 0 && is_allowed_weight_class(fight->weight_class)) {
            (*count)++;
        }
    }
    
    fclose(f);
    printf("Loaded %d valid UFC fights from %s\n", *count, path);
    return 0;
}

/* Feature engineering: compute deltas and derived features */
static void compute_features(UFCFight *fight, double *features, MatchContext *ctx) {
    /* Basic deltas (fighter1 - fighter2) */
    features[FEAT_HEIGHT_DELTA] = fight->f1_height - fight->f2_height;
    features[FEAT_REACH_DELTA] = fight->f1_reach - fight->f2_reach;
    features[FEAT_AGE_DELTA] = fight->f1_age - fight->f2_age;
    features[FEAT_SIG_STRIKE_PM_DELTA] = fight->f1_sig_strikes_pm - fight->f2_sig_strikes_pm;
    features[FEAT_SIG_STRIKE_ACC_DELTA] = fight->f1_sig_strikes_acc - fight->f2_sig_strikes_acc;
    features[FEAT_SIG_STRIKE_ABS_DELTA] = fight->f1_sig_strikes_abs - fight->f2_sig_strikes_abs;
    features[FEAT_SIG_STRIKE_DEF_DELTA] = fight->f1_sig_strikes_def - fight->f2_sig_strikes_def;
    features[FEAT_TAKEDOWN_AVG_DELTA] = fight->f1_takedown_avg - fight->f2_takedown_avg;
    features[FEAT_TAKEDOWN_ACC_DELTA] = fight->f1_takedown_acc - fight->f2_takedown_acc;
    features[FEAT_TAKEDOWN_DEF_DELTA] = fight->f1_takedown_def - fight->f2_takedown_def;
    features[FEAT_SUB_AVG_DELTA] = fight->f1_sub_avg - fight->f2_sub_avg;
    features[FEAT_WEIGHT_DELTA] = fight->f1_weight - fight->f2_weight;
    
    /* Derived features */
    /* Striking advantage: net striking effectiveness */
    double f1_strike_net = fight->f1_sig_strikes_pm - fight->f1_sig_strikes_abs;
    double f2_strike_net = fight->f2_sig_strikes_pm - fight->f2_sig_strikes_abs;
    features[FEAT_STRIKING_ADVANTAGE] = f1_strike_net - f2_strike_net;
    
    /* Grappling composite score */
    double f1_grapple = fight->f1_takedown_avg * fight->f1_takedown_acc + fight->f1_sub_avg;
    double f2_grapple = fight->f2_takedown_avg * fight->f2_takedown_acc + fight->f2_sub_avg;
    features[FEAT_GRAPPLING_SCORE] = f1_grapple - f2_grapple;

    FighterRecord *r1 = NULL;
    FighterRecord *r2 = NULL;
    HeadToHeadRecord *h = NULL;
    if (ctx) {
        r1 = get_or_add_fighter(ctx, fight->fighter1, fight->weight_class);
        r2 = get_or_add_fighter(ctx, fight->fighter2, fight->weight_class);
        h = get_or_add_h2h(ctx, fight->fighter1, fight->fighter2, fight->weight_class);
    }

    double wr1 = prior_win_rate(r1);
    double wr2 = prior_win_rate(r2);
    int wins1 = r1 ? r1->wins : 0;
    int wins2 = r2 ? r2->wins : 0;
    int fights1 = r1 ? r1->total : 0;
    int fights2 = r2 ? r2->total : 0;
    double wscore1 = weighted_score(r1);
    double wscore2 = weighted_score(r2);

    features[FEAT_WIN_RATE_DELTA] = wr1 - wr2;
    features[FEAT_TOTAL_WINS_DELTA] = (double)(wins1 - wins2);
    features[FEAT_TOTAL_FIGHTS_DELTA] = (double)(fights1 - fights2);
    features[FEAT_WEIGHTED_SCORE_DELTA] = wscore1 - wscore2;

    int style1 = fight->f1_sub_avg >= 1.0 ? 1 : 0;
    int style2 = fight->f2_sub_avg >= 1.0 ? 1 : 0;
    features[FEAT_SUB_STYLE_DELTA] = (double)(style1 - style2);

    features[FEAT_H2H_DELTA] = head_to_head_bias(h, fight->fighter1, fight->fighter2);
}

/* Compute normalization statistics (mean and std dev) */
static void compute_normalization(double (*raw)[INPUT_SIZE], int count, Model *m) {
    double sum[INPUT_SIZE] = {0};
    double sq_sum[INPUT_SIZE] = {0};

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum[j] += raw[i][j];
            sq_sum[j] += raw[i][j] * raw[i][j];
        }
    }
    
    for (int j = 0; j < INPUT_SIZE; j++) {
        m->feat_mean[j] = sum[j] / (double)count;
        double variance = (sq_sum[j] / (double)count) - (m->feat_mean[j] * m->feat_mean[j]);
        m->feat_std[j] = sqrt(variance > 0 ? variance : 1e-8);
        if (m->feat_std[j] < 1e-8) m->feat_std[j] = 1.0;  /* prevent division by zero */
    }
}

/* Normalize features using computed statistics */
static void normalize_features(double *features, Model *m, long double *normalized) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        normalized[i] = (long double)((features[i] - m->feat_mean[i]) / m->feat_std[i]);
    }
}

/* Forward pass */
void forward(Model *m, long double *in) {
    for (int j = 0; j < HIDDEN_1_SIZE; j++) {
        long double sum = m->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += in[i] * m->w1[i][j];
        }
        m->h1[j] = tanh_act(sum);
    }
    
    for (int j = 0; j < HIDDEN_2_SIZE; j++) {
        long double sum = m->b2[j];
        for (int i = 0; i < HIDDEN_1_SIZE; i++) {
            sum += m->h1[i] * m->w2[i][j];
        }
        m->h2[j] = tanh_act(sum);
    }
    
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        long double sum = m->b3[k];
        for (int j = 0; j < HIDDEN_2_SIZE; j++) {
            sum += m->h2[j] * m->w3[j][k];
        }
        m->output[k] = sigmoid(sum);
    }
}

/* Backward pass with gradient descent */
void backward(Model *m, long double *in, long double target, long double lr, long double mom) {
    /* Output layer delta */
    long double out_delta = (m->output[0] - target) * sigmoid_deriv(m->output[0]);
    
    /* Hidden layer 2 deltas */
    long double h2_delta[HIDDEN_2_SIZE];
    for (int j = 0; j < HIDDEN_2_SIZE; j++) {
        h2_delta[j] = out_delta * m->w3[j][0] * tanh_deriv(m->h2[j]);
    }
    
    /* Hidden layer 1 deltas */
    long double h1_delta[HIDDEN_1_SIZE];
    for (int j = 0; j < HIDDEN_1_SIZE; j++) {
        long double err = 0;
        for (int k = 0; k < HIDDEN_2_SIZE; k++) {
            err += h2_delta[k] * m->w2[j][k];
        }
        h1_delta[j] = err * tanh_deriv(m->h1[j]);
    }
    
    /* Update output layer weights */
    for (int j = 0; j < HIDDEN_2_SIZE; j++) {
        m->v_w3[j][0] = mom * m->v_w3[j][0] - lr * out_delta * m->h2[j];
        m->w3[j][0] += m->v_w3[j][0];
    }
    m->v_b3[0] = mom * m->v_b3[0] - lr * out_delta;
    m->b3[0] += m->v_b3[0];
    
    /* Update hidden layer 2 weights */
    for (int j = 0; j < HIDDEN_2_SIZE; j++) {
        for (int i = 0; i < HIDDEN_1_SIZE; i++) {
            m->v_w2[i][j] = mom * m->v_w2[i][j] - lr * h2_delta[j] * m->h1[i];
            m->w2[i][j] += m->v_w2[i][j];
        }
        m->v_b2[j] = mom * m->v_b2[j] - lr * h2_delta[j];
        m->b2[j] += m->v_b2[j];
    }
    
    /* Update hidden layer 1 weights */
    for (int j = 0; j < HIDDEN_1_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            m->v_w1[i][j] = mom * m->v_w1[i][j] - lr * h1_delta[j] * in[i];
            m->w1[i][j] += m->v_w1[i][j];
        }
        m->v_b1[j] = mom * m->v_b1[j] - lr * h1_delta[j];
        m->b1[j] += m->v_b1[j];
    }
}

/* Initialize model weights */
void init_model(Model *m) {
    memset(m, 0, sizeof(Model));
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_1_SIZE; j++) {
            m->w1[i][j] = he_init(INPUT_SIZE);
        }
    }
    
    for (int i = 0; i < HIDDEN_1_SIZE; i++) {
        for (int j = 0; j < HIDDEN_2_SIZE; j++) {
            m->w2[i][j] = he_init(HIDDEN_1_SIZE);
        }
    }
    
    for (int i = 0; i < HIDDEN_2_SIZE; i++) {
        m->w3[i][0] = he_init(HIDDEN_2_SIZE);
    }
}

/* Train the model */
void train(Model *m, UFCFight *fights, int num_samples) {
    printf("\n=== Training UFC Winner Prediction Model ===\n");
    printf("Samples: %d\n", num_samples);
    printf("Architecture: %d -> %d -> %d -> %d\n", 
           INPUT_SIZE, HIDDEN_1_SIZE, HIDDEN_2_SIZE, OUTPUT_SIZE);
    
    qsort(fights, (size_t)num_samples, sizeof(UFCFight), compare_fight_date);

    int train_base_count = (num_samples * 8) / 10;
    if (train_base_count < 1) {
        train_base_count = 1;
    }
    if (train_base_count >= num_samples) {
        train_base_count = num_samples - 1;
    }
    int val_count = num_samples - train_base_count;
    if (val_count < 1) {
        fprintf(stderr, "not enough samples for validation split\n");
        return;
    }
    printf("Split: train=%d  validation=%d (chronological)\n", train_base_count, val_count);

    /* Prepare contextual raw features using only training history for validation rows */
    double (*raw_features)[INPUT_SIZE] = malloc((size_t)num_samples * sizeof(*raw_features));
    if (!raw_features) {
        fprintf(stderr, "failed to allocate raw feature matrix\n");
        return;
    }

    long double *labels = malloc((size_t)num_samples * sizeof(*labels));
    if (!labels) {
        fprintf(stderr, "failed to allocate labels\n");
        free(raw_features);
        return;
    }

    MatchContext train_ctx;
    init_context(&train_ctx);
    for (int i = 0; i < num_samples; i++) {
        compute_features(&fights[i], raw_features[i], &train_ctx);
        labels[i] = (long double)fights[i].label;
        if (i < train_base_count) {
            update_context_with_result(&train_ctx, &fights[i]);
        }
    }
    free_context(&train_ctx);

    int augmented_samples = train_base_count * 2;
    double (*aug_raw)[INPUT_SIZE] = malloc((size_t)augmented_samples * sizeof(*aug_raw));
    long double *aug_labels = malloc((size_t)augmented_samples * sizeof(*aug_labels));
    if (!aug_raw || !aug_labels) {
        fprintf(stderr, "failed to allocate augmented training buffers\n");
        free(raw_features);
        free(labels);
        free(aug_raw);
        free(aug_labels);
        return;
    }

    for (int i = 0; i < train_base_count; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            aug_raw[i][j] = raw_features[i][j];
            aug_raw[i + train_base_count][j] = -raw_features[i][j];
        }
        aug_labels[i] = labels[i];
        aug_labels[i + train_base_count] = 1.0L - labels[i];
    }

    /* Compute normalization statistics from enriched and mirrored features */
    compute_normalization(aug_raw, augmented_samples, m);

    /* Prepare normalized training data */
    long double (*data)[INPUT_SIZE] = malloc((size_t)augmented_samples * sizeof(*data));
    if (!data) {
        fprintf(stderr, "failed to allocate normalized feature matrix\n");
        free(raw_features);
        free(labels);
        free(aug_raw);
        free(aug_labels);
        return;
    }

    for (int i = 0; i < augmented_samples; i++) {
        normalize_features(aug_raw[i], m, data[i]);
    }
    free(aug_raw);

    /* Prepare normalized validation data (non-augmented, chronological holdout) */
    long double (*val_data)[INPUT_SIZE] = malloc((size_t)val_count * sizeof(*val_data));
    long double *val_labels = malloc((size_t)val_count * sizeof(*val_labels));
    if (!val_data || !val_labels) {
        fprintf(stderr, "failed to allocate validation buffers\n");
        free(raw_features);
        free(labels);
        free(data);
        free(aug_labels);
        free(val_data);
        free(val_labels);
        return;
    }
    for (int i = 0; i < val_count; i++) {
        normalize_features(raw_features[train_base_count + i], m, val_data[i]);
        val_labels[i] = labels[train_base_count + i];
    }

    free(raw_features);
    free(labels);
    labels = aug_labels;
    num_samples = augmented_samples;
    int train_count = num_samples;
    
    /* Training hyperparameters */
    const int max_epochs = 500;
    const long double initial_lr = 0.01L;
    const long double momentum = 0.9L;
    const long double min_lr = 0.0001L;
    
    printf("\nTraining for up to %d epochs...\n", max_epochs);
    
    long double best_val_acc = 0.0L;
    int patience = 50;
    int no_improve = 0;
    
    for (int epoch = 0; epoch < max_epochs && keep_running; epoch++) {
        /* Learning rate decay */
        long double lr = initial_lr * powl(0.95L, (long double)epoch / 20.0L);
        if (lr < min_lr) lr = min_lr;
        
        /* Shuffle training data */
        for (int i = train_count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            /* Swap data[i] and data[j] */
            long double tmp_data[INPUT_SIZE];
            memcpy(tmp_data, data[i], sizeof(tmp_data));
            memcpy(data[i], data[j], sizeof(tmp_data));
            memcpy(data[j], tmp_data, sizeof(tmp_data));
            /* Swap labels */
            long double tmp_label = labels[i];
            labels[i] = labels[j];
            labels[j] = tmp_label;
        }
        
        /* Train on all samples */
        long double total_loss = 0.0L;
        int correct = 0;
        
        for (int i = 0; i < train_count; i++) {
            forward(m, data[i]);
            long double pred = m->output[0];
            long double target = labels[i];
            
            /* Cross-entropy loss */
            total_loss += -target * logl(pred + 1e-15L) - (1.0L - target) * logl(1.0L - pred + 1e-15L);
            
            /* Accuracy */
            int pred_class = pred >= 0.5L ? 1 : 0;
            int true_class = (int)target;
            if (pred_class == true_class) correct++;
            
            backward(m, data[i], target, lr, momentum);
        }
        
        long double avg_loss = total_loss / (long double)train_count;
        long double train_acc = (long double)correct / (long double)train_count;

        long double val_loss = 0.0L;
        int val_correct = 0;
        for (int i = 0; i < val_count; i++) {
            forward(m, val_data[i]);
            long double pred = m->output[0];
            long double target = val_labels[i];
            val_loss += -target * logl(pred + 1e-15L) - (1.0L - target) * logl(1.0L - pred + 1e-15L);
            int pred_class = pred >= 0.5L ? 1 : 0;
            int true_class = (int)target;
            if (pred_class == true_class) {
                val_correct++;
            }
        }
        long double avg_val_loss = val_loss / (long double)val_count;
        long double val_acc = (long double)val_correct / (long double)val_count;
        
        /* Print progress every 10 epochs */
        if (epoch % 10 == 0 || epoch == max_epochs - 1) {
            printf("Epoch %3d/%d  TrainLoss: %.6Lf  TrainAcc: %.2Lf%%  ValLoss: %.6Lf  ValAcc: %.2Lf%%  LR: %.6Lf\n",
                   epoch + 1, max_epochs, avg_loss, train_acc * 100.0L,
                   avg_val_loss, val_acc * 100.0L, lr);
        }
        
        /* Early stopping on validation accuracy */
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= patience) {
                printf("\nEarly stopping at epoch %d (best validation accuracy: %.2Lf%%)\n",
                       epoch + 1, best_val_acc * 100.0L);
                break;
            }
        }
    }
    
    printf("\n=== Training Complete ===\n");
    printf("Best validation accuracy: %.2Lf%%\n", best_val_acc * 100.0L);
    
    m->num_trained_samples = train_base_count;
    
    free(data);
    free(labels);
    free(val_data);
    free(val_labels);
}

void evaluate_model(Model *m, UFCFight *fights, int num_samples) {
    qsort(fights, (size_t)num_samples, sizeof(UFCFight), compare_fight_date);

    int train_base_count = (num_samples * 8) / 10;
    if (train_base_count < 1) {
        train_base_count = 1;
    }
    if (train_base_count >= num_samples) {
        train_base_count = num_samples - 1;
    }
    int val_count = num_samples - train_base_count;
    if (val_count < 1) {
        fprintf(stderr, "not enough samples for evaluation split\n");
        return;
    }

    double (*raw_features)[INPUT_SIZE] = malloc((size_t)num_samples * sizeof(*raw_features));
    long double *labels = malloc((size_t)num_samples * sizeof(*labels));
    if (!raw_features || !labels) {
        fprintf(stderr, "failed to allocate evaluation buffers\n");
        free(raw_features);
        free(labels);
        return;
    }

    MatchContext ctx;
    init_context(&ctx);
    for (int i = 0; i < num_samples; i++) {
        compute_features(&fights[i], raw_features[i], &ctx);
        labels[i] = (long double)fights[i].label;
        if (i < train_base_count) {
            update_context_with_result(&ctx, &fights[i]);
        }
    }
    free_context(&ctx);

    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    long double val_loss = 0.0L;

    long double *val_probs = malloc((size_t)val_count * sizeof(*val_probs));
    int *val_true = malloc((size_t)val_count * sizeof(*val_true));
    if (!val_probs || !val_true) {
        fprintf(stderr, "failed to allocate probability buffers\n");
        free(raw_features);
        free(labels);
        free(val_probs);
        free(val_true);
        return;
    }

    ClassStats *class_stats = NULL;
    int class_count = 0;
    int class_cap = 0;

    for (int i = train_base_count; i < num_samples; i++) {
        int v_idx = i - train_base_count;
        long double normalized[INPUT_SIZE];
        normalize_features(raw_features[i], m, normalized);
        forward(m, normalized);

        long double pred = m->output[0];
        int true_class = (int)labels[i];

        val_probs[v_idx] = pred;
        val_true[v_idx] = true_class;

        val_loss += -labels[i] * logl(pred + 1e-15L) - (1.0L - labels[i]) * logl(1.0L - pred + 1e-15L);
    }

    long double best_threshold = 0.5L;
    long double best_bal_acc = -1.0L;
    for (int step = 20; step <= 80; step++) {
        long double threshold = (long double)step / 100.0L;
        int stp = 0, stn = 0, sfp = 0, sfn = 0;
        for (int i = 0; i < val_count; i++) {
            int pred_class = val_probs[i] >= threshold ? 1 : 0;
            int true_class = val_true[i];
            if (pred_class == 1 && true_class == 1) stp++;
            else if (pred_class == 0 && true_class == 0) stn++;
            else if (pred_class == 1 && true_class == 0) sfp++;
            else if (pred_class == 0 && true_class == 1) sfn++;
        }
        long double tpr = (stp + sfn) > 0 ? (long double)stp / (long double)(stp + sfn) : 0.0L;
        long double tnr = (stn + sfp) > 0 ? (long double)stn / (long double)(stn + sfp) : 0.0L;
        long double bal_acc = 0.5L * (tpr + tnr);
        if (bal_acc > best_bal_acc) {
            best_bal_acc = bal_acc;
            best_threshold = threshold;
        }
    }

    for (int i = 0; i < val_count; i++) {
        int pred_class = val_probs[i] >= best_threshold ? 1 : 0;
        int true_class = val_true[i];
        if (pred_class == 1 && true_class == 1) tp++;
        else if (pred_class == 0 && true_class == 0) tn++;
        else if (pred_class == 1 && true_class == 0) fp++;
        else if (pred_class == 0 && true_class == 1) fn++;

        ClassStats *bucket = get_or_add_class_stats(&class_stats, &class_count, &class_cap, fights[train_base_count + i].weight_class);
        if (bucket) {
            bucket->total++;
            if (pred_class == true_class) {
                bucket->correct++;
            }
        }
    }

    int total = tp + tn + fp + fn;
    long double accuracy = total > 0 ? (long double)(tp + tn) / (long double)total : 0.0L;
    long double precision = (tp + fp) > 0 ? (long double)tp / (long double)(tp + fp) : 0.0L;
    long double recall = (tp + fn) > 0 ? (long double)tp / (long double)(tp + fn) : 0.0L;
    long double f1 = (precision + recall) > 0 ? 2.0L * precision * recall / (precision + recall) : 0.0L;
    long double specificity = (tn + fp) > 0 ? (long double)tn / (long double)(tn + fp) : 0.0L;
    long double balanced_accuracy = 0.5L * (recall + specificity);
    long double avg_val_loss = val_count > 0 ? val_loss / (long double)val_count : 0.0L;

    printf("\n=== Evaluation : Chronological Holdout ===\n");
    printf("Validation samples : %d\n", val_count);
    printf("Loss : %.6Lf\n", avg_val_loss);
    printf("Best threshold : %.2Lf\n", best_threshold);
    printf("Accuracy : %.2Lf%%\n", accuracy * 100.0L);
    printf("Precision : %.2Lf%%\n", precision * 100.0L);
    printf("Recall : %.2Lf%%\n", recall * 100.0L);
    printf("Specificity : %.2Lf%%\n", specificity * 100.0L);
    printf("Balanced Accuracy : %.2Lf%%\n", balanced_accuracy * 100.0L);
    printf("F1 : %.2Lf%%\n", f1 * 100.0L);

    printf("\nConfusion Matrix :\n");
    printf("  TP=%d  FP=%d\n", tp, fp);
    printf("  FN=%d  TN=%d\n", fn, tn);

    printf("\nPer-Weight-Class Accuracy :\n");
    for (int i = 0; i < class_count; i++) {
        long double class_acc = class_stats[i].total > 0
            ? (long double)class_stats[i].correct / (long double)class_stats[i].total
            : 0.0L;
        printf("  %s : %.2Lf%% (%d/%d)\n",
               class_stats[i].weight_class,
               class_acc * 100.0L,
               class_stats[i].correct,
               class_stats[i].total);
    }

    free(class_stats);
    free(raw_features);
    free(labels);
    free(val_probs);
    free(val_true);
}

void analyze_matchup(UFCFight *fights, int num_fights) {
    char weight_class[64] = {0};
    char fighter_a[64] = {0};
    char fighter_b[64] = {0};
    char buffer[256];

    printf("\n=== Matchup Analysis ===\n");
    printf("Weight class: ");
    fflush(stdout);
    if (!fgets(buffer, sizeof(buffer), stdin)) return;
    buffer[strcspn(buffer, "\n")] = '\0';
    strncpy(weight_class, buffer, sizeof(weight_class) - 1);
    if (!is_allowed_weight_class(weight_class)) {
        printf("Invalid weight class input\n");
        return;
    }

    printf("Fighter A name: ");
    fflush(stdout);
    if (!fgets(buffer, sizeof(buffer), stdin)) return;
    buffer[strcspn(buffer, "\n")] = '\0';
    strncpy(fighter_a, buffer, sizeof(fighter_a) - 1);

    printf("Fighter B name: ");
    fflush(stdout);
    if (!fgets(buffer, sizeof(buffer), stdin)) return;
    buffer[strcspn(buffer, "\n")] = '\0';
    strncpy(fighter_b, buffer, sizeof(fighter_b) - 1);

    if (fighter_a[0] == '\0' || fighter_b[0] == '\0' || weight_class[0] == '\0') {
        printf("Invalid input\n");
        return;
    }

    UFCFight query = {0};
    strncpy(query.weight_class, weight_class, sizeof(query.weight_class) - 1);
    strncpy(query.fighter1, fighter_a, sizeof(query.fighter1) - 1);
    strncpy(query.fighter2, fighter_b, sizeof(query.fighter2) - 1);

    if (load_latest_fighter_snapshot(fights, num_fights, weight_class, fighter_a, &query, 1) != 0) {
        printf("No stats found for fighter A in this weight class\n");
        return;
    }
    if (load_latest_fighter_snapshot(fights, num_fights, weight_class, fighter_b, &query, 2) != 0) {
        printf("No stats found for fighter B in this weight class\n");
        return;
    }

    MatchContext ctx;
    init_context(&ctx);
    for (int i = 0; i < num_fights; i++) {
        if (strcmp(fights[i].weight_class, weight_class) == 0) {
            update_context_with_result(&ctx, &fights[i]);
        }
    }

    char model_path[256];
    if (build_class_model_path(weight_class, model_path, sizeof(model_path)) != 0) {
        free_context(&ctx);
        printf("Could not build model path for this class\n");
        return;
    }

    Model model = {0};
    if (load_model(&model, model_path) != 0) {
        free_context(&ctx);
        printf("Class model not found: %s\n", model_path);
        printf("Run class training first (make train)\n");
        return;
    }

    double features[INPUT_SIZE];
    long double normalized[INPUT_SIZE];
    compute_features(&query, features, &ctx);
    normalize_features(features, &model, normalized);
    forward(&model, normalized);

    long double p_a = model.output[0];
    long double p_b = 1.0L - p_a;

    print_side_by_side_stats(&query);

    int a_h2h_wins = 0;
    int b_h2h_wins = 0;
    int h2h_total = 0;
    for (int i = 0; i < num_fights; i++) {
        if (strcmp(fights[i].weight_class, weight_class) != 0) {
            continue;
        }
        int direct = 0;
        if (strcmp(fights[i].fighter1, fighter_a) == 0 && strcmp(fights[i].fighter2, fighter_b) == 0) {
            direct = 1;
        } else if (strcmp(fights[i].fighter1, fighter_b) == 0 && strcmp(fights[i].fighter2, fighter_a) == 0) {
            direct = 1;
        }
        if (!direct) {
            continue;
        }

        h2h_total++;
        if (fights[i].label == 1 && strcmp(fights[i].fighter1, fighter_a) == 0) {
            a_h2h_wins++;
        } else if (fights[i].label == 0 && strcmp(fights[i].fighter2, fighter_a) == 0) {
            a_h2h_wins++;
        } else {
            b_h2h_wins++;
        }
    }

    const FighterRecord *a_record = find_fighter_record(&ctx, fighter_a, weight_class);
    const FighterRecord *b_record = find_fighter_record(&ctx, fighter_b, weight_class);

    printf("\n--- Model Prediction ---\n");
    printf("Weight class : %s\n", weight_class);
    printf("P(%s wins) : %.2Lf%%\n", fighter_a, p_a * 100.0L);
    printf("P(%s wins) : %.2Lf%%\n", fighter_b, p_b * 100.0L);

    printf("\n--- Reality from Data ---\n");
    if (a_record) {
        printf("%s class record : %d-%d (%d fights)\n", fighter_a, a_record->wins, a_record->losses, a_record->total);
    } else {
        printf("%s class record : not found\n", fighter_a);
    }
    if (b_record) {
        printf("%s class record : %d-%d (%d fights)\n", fighter_b, b_record->wins, b_record->losses, b_record->total);
    } else {
        printf("%s class record : not found\n", fighter_b);
    }

    if (h2h_total > 0) {
        long double empirical_a = (long double)a_h2h_wins / (long double)h2h_total;
        long double empirical_b = (long double)b_h2h_wins / (long double)h2h_total;
        printf("Head-to-head : %s %d wins, %s %d wins (%d fights)\n",
               fighter_a, a_h2h_wins, fighter_b, b_h2h_wins, h2h_total);
        printf("Empirical odds : %s %.2Lf%%, %s %.2Lf%%\n",
               fighter_a, empirical_a * 100.0L, fighter_b, empirical_b * 100.0L);
        printf("Model vs empirical delta : %.2Lf percentage points\n", fabsl((p_a - empirical_a) * 100.0L));
    } else {
        printf("Head-to-head : no direct fights found in dataset\n");
        printf("Empirical odds : unavailable for this matchup\n");
    }

    free_context(&ctx);
}

int train_models_by_class(UFCFight *fights, int num_fights) {
    WeightClassBucket *buckets = NULL;
    int bucket_count = 0;
    int bucket_cap = 0;

    for (int i = 0; i < num_fights; i++) {
        const char *wc = fights[i].weight_class[0] ? fights[i].weight_class : "Unknown";
        WeightClassBucket *bucket = get_or_add_bucket(&buckets, &bucket_count, &bucket_cap, wc);
        if (!bucket) {
            fprintf(stderr, "failed to allocate class bucket\n");
            free_buckets(buckets, bucket_count);
            return -1;
        }
        if (bucket_push_fight(bucket, &fights[i]) != 0) {
            fprintf(stderr, "failed to append fight into class bucket\n");
            free_buckets(buckets, bucket_count);
            return -1;
        }
    }

    printf("\n=== Class-Specific Training ===\n");
    printf("Weight classes found : %d\n", bucket_count);

    for (int i = 0; i < bucket_count; i++) {
        if (buckets[i].count < 20) {
            printf("Skipping %s : only %d fights\n", buckets[i].weight_class, buckets[i].count);
            continue;
        }

        printf("\nTraining class : %s (%d fights)\n", buckets[i].weight_class, buckets[i].count);
        Model class_model = {0};
        init_model(&class_model);
        train(&class_model, buckets[i].fights, buckets[i].count);

        char class_model_path[256];
        if (build_class_model_path(buckets[i].weight_class, class_model_path, sizeof(class_model_path)) != 0) {
            fprintf(stderr, "failed to build model path for class %s\n", buckets[i].weight_class);
            free_buckets(buckets, bucket_count);
            return -1;
        }
        if (save_model(&class_model, class_model_path) != 0) {
            fprintf(stderr, "failed to save model for class %s\n", buckets[i].weight_class);
            free_buckets(buckets, bucket_count);
            return -1;
        }
    }

    free_buckets(buckets, bucket_count);
    return 0;
}

/* Save model to file */
int save_model(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(m, sizeof(Model), 1, f);
    fclose(f);
    printf("Model saved to %s\n", path);
    return 0;
}

/* Load model from file */
int load_model(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    
    if (sz < (long)sizeof(Model)) {
        fclose(f);
        return -1;
    }
    
    if (fread(m, sizeof(Model), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    fclose(f);
    printf("Model loaded from %s\n", path);
    return 0;
}

/* Interactive prediction mode */
void predict_interactive(Model *m, UFCFight *fights, int num_fights, int future_mode) {
    printf("\n=== UFC Fight Predictor ===\n");
    if (!future_mode) {
        printf("Mode : historical-aware (uses prior records and head-to-head if available)\n");
    } else {
        printf("Mode : future-fight (uses damped other-fight history, ignores direct head-to-head)\n");
    }
    printf("Enter weight class (or auto), fighter names, and statistics to predict winner probability\n");
    printf("(or 'q' to quit)\n\n");

    MatchContext pred_ctx;
    init_context(&pred_ctx);
    for (int i = 0; i < num_fights; i++) {
        update_context_with_result(&pred_ctx, &fights[i]);
    }
    
    char buffer[1024];
    while (keep_running) {
        UFCFight fight = {0};
        char requested_weight_class[64] = {0};

        printf("Weight class (or auto)\n> ");
        fflush(stdout);
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) break;
        if (buffer[0] == 'q') break;
        buffer[strcspn(buffer, "\n")] = '\0';
        strncpy(requested_weight_class, buffer, sizeof(requested_weight_class) - 1);
        printf("\n");

        printf("Fighter 1 name\n> ");
        fflush(stdout);
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) break;
        if (buffer[0] == 'q') break;
        buffer[strcspn(buffer, "\n")] = '\0';
        strncpy(fight.fighter1, buffer, sizeof(fight.fighter1) - 1);
        printf("\n");

        printf("Fighter 2 name\n> ");
        fflush(stdout);
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) break;
        if (buffer[0] == 'q') break;
        buffer[strcspn(buffer, "\n")] = '\0';
        strncpy(fight.fighter2, buffer, sizeof(fight.fighter2) - 1);
        printf("\n");

        int f1_in_history = fighter_exists_in_history(fights, num_fights, fight.fighter1);
        int f2_in_history = fighter_exists_in_history(fights, num_fights, fight.fighter2);
        if (!f1_in_history || !f2_in_history) {
            printf("Historical data check failed : ");
            if (!f1_in_history) {
                printf("%s not found", fight.fighter1);
            }
            if (!f1_in_history && !f2_in_history) {
                printf("; ");
            }
            if (!f2_in_history) {
                printf("%s not found", fight.fighter2);
            }
            printf("\n");
            printf("Predicted winner: Unsure (missing historical data)\n\n");
            if (!isatty(fileno(stdin))) {
                break;
            }
            continue;
        }

        if (requested_weight_class[0] == '\0' || name_equals_ci(requested_weight_class, "auto")) {
            int infer_status = infer_latest_weight_class(
                fights,
                num_fights,
                fight.fighter1,
                fight.fighter2,
                fight.weight_class,
                sizeof(fight.weight_class)
            );
            if (infer_status == 0) {
                printf("Could not infer weight class from recent fights. Enter a weight class manually.\n\n");
                if (!isatty(fileno(stdin))) {
                    break;
                }
                continue;
            }
            printf("Auto-selected most recent weight class : %s\n", fight.weight_class);
            if (infer_status == 2) {
                printf("Fighters have different latest classes : using the most recent one between them\n");
            }
        } else {
            strncpy(fight.weight_class, requested_weight_class, sizeof(fight.weight_class) - 1);
        }

        if (!is_allowed_weight_class(fight.weight_class)) {
            printf("Invalid weight class input\n\n");
            if (!isatty(fileno(stdin))) {
                break;
            }
            continue;
        }

        double dict_stats[12];
        int f1_loaded = lookup_fighter_stats_from_dict(FIGHTER_STATS_DICT_PATH, fight.weight_class, fight.fighter1, dict_stats);
        if (f1_loaded) apply_fighter_stats(&fight, 1, dict_stats);
        int f2_loaded = lookup_fighter_stats_from_dict(FIGHTER_STATS_DICT_PATH, fight.weight_class, fight.fighter2, dict_stats);
        if (f2_loaded) apply_fighter_stats(&fight, 2, dict_stats);

        if (f1_loaded || f2_loaded) {
            printf("Auto-loaded from JSON dictionary :");
            if (f1_loaded) printf(" fighter1");
            if (f2_loaded) printf(" fighter2");
            printf("\n");
        }

        if (!f1_loaded) {
            int r = prompt_fighter_stats("Fighter 1", &fight, 1, buffer, sizeof(buffer));
            if (r == 0) break;
            if (r < 0) {
                printf("Invalid input format\n");
                continue;
            }
            printf("\n");
        }

        if (!f2_loaded) {
            int r = prompt_fighter_stats("Fighter 2", &fight, 2, buffer, sizeof(buffer));
            if (r == 0) break;
            if (r < 0) {
                printf("Invalid input format\n");
                continue;
            }
            printf("\n");
        }
        
        Model active_model = *m;
        char class_model_path[256];
        if (build_class_model_path(fight.weight_class, class_model_path, sizeof(class_model_path)) == 0) {
            Model class_model = {0};
            if (load_model(&class_model, class_model_path) == 0) {
                active_model = class_model;
            } else {
                printf("Class model not found for %s : using current loaded model\n", fight.weight_class);
            }
        }

         print_side_by_side_stats(&fight);

        if (fighter_stats_identical(&fight, 1e-12)) {
            printf("\n--- Prediction ---\n");
            printf("Fighter 1 win probability: 50.00%%\n");
            printf("Fighter 2 win probability: 50.00%%\n");
            printf("Predicted winner: Unable to predict winner (identical input stats)\n\n");
            continue;
        }

        double features[INPUT_SIZE];
        long double normalized[INPUT_SIZE];
        compute_features(&fight, features, &pred_ctx);
        if (future_mode) {
            const double context_scale = 0.20;
            const double h2h_scale = 0.05;
            features[FEAT_WIN_RATE_DELTA] *= context_scale;
            features[FEAT_TOTAL_WINS_DELTA] *= context_scale;
            features[FEAT_TOTAL_FIGHTS_DELTA] *= context_scale;
            features[FEAT_WEIGHTED_SCORE_DELTA] *= context_scale;
            features[FEAT_H2H_DELTA] *= h2h_scale;
        }
        normalize_features(features, &active_model, normalized);
        
        forward(&active_model, normalized);
        
        long double prob_f1_wins = active_model.output[0];
        if (future_mode) {
            prob_f1_wins = temperature_scale_probability(prob_f1_wins, 4.0L);
        }
        long double prob_f2_wins = 1.0L - prob_f1_wins;
        
        printf("\n--- Prediction ---\n");
        printf("Fighter 1 win probability: %.2Lf%%\n", prob_f1_wins * 100.0L);
        printf("Fighter 2 win probability: %.2Lf%%\n", prob_f2_wins * 100.0L);
        printf("Prior win-rate delta: %.3f\n", features[FEAT_WIN_RATE_DELTA]);
        printf("Prior total-wins delta: %.0f\n", features[FEAT_TOTAL_WINS_DELTA]);
        if (future_mode) {
            printf("Prior head-to-head bias: %.3f (low-weight in future-fight mode)\n", features[FEAT_H2H_DELTA]);
        } else {
            printf("Prior head-to-head bias: %.3f\n", features[FEAT_H2H_DELTA]);
        }
        const long double decision_threshold = 0.95L;
        if (prob_f1_wins >= decision_threshold) {
            printf("Predicted winner: %s\n\n", fight.fighter1);
        } else if (prob_f2_wins >= decision_threshold) {
            printf("Predicted winner: %s\n\n", fight.fighter2);
        } else {
            printf("Predicted winner: Unsure (confidence below 95%%)\n\n");
        }
    }

    free_context(&pred_ctx);
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    
    /* Install SIGINT handler */
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  UFC Fight Winner Prediction Model    ║\n");
    printf("║  Neural Network in C99                 ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    const char *data_path = "../data/ufc_fights_full_with_odds.csv";
    const char *model_path = "ufc_model.bin";
    int flag_load = 0;
    int flag_predict = 0;
    int flag_eval = 0;
    int flag_global = 0;
    int flag_matchup = 0;
    int flag_predict_future = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0) {
            flag_load = 1;
        } else if (strcmp(argv[i], "--predict") == 0) {
            flag_predict = 1;
            flag_predict_future = 1;
        } else if (strcmp(argv[i], "--predict-future") == 0) {
            flag_predict = 1;
            flag_predict_future = 1;
        } else if (strcmp(argv[i], "--predict-historical") == 0) {
            flag_predict = 1;
            flag_predict_future = 0;
        } else if (strcmp(argv[i], "--eval") == 0) {
            flag_eval = 1;
        } else if (strcmp(argv[i], "--global") == 0) {
            flag_global = 1;
        } else if (strcmp(argv[i], "--matchup") == 0) {
            flag_matchup = 1;
        }
    }
    
    /* Load UFC dataset */
    UFCFight *fights = NULL;
    int num_fights = 0;
    
    if (load_ufc_data(data_path, &fights, &num_fights) < 0) {
        fprintf(stderr, "Failed to load UFC data\n");
        return 1;
    }
    
    Model model = {0};

    if (!flag_global && !flag_load) {
        if (train_models_by_class(fights, num_fights) != 0) {
            fprintf(stderr, "class-specific training failed\n");
            free(fights);
            return 1;
        }
        if (flag_predict) {
            Model fallback_model = {0};
            if (load_model(&fallback_model, model_path) != 0) {
                init_model(&fallback_model);
                train(&fallback_model, fights, num_fights);
                save_model(&fallback_model, model_path);
            }
            model = fallback_model;
        }
    } else if (!flag_global && flag_load) {
        if (load_model(&model, model_path) != 0) {
            memset(&model, 0, sizeof(model));
            printf("Global fallback model not found : class-specific models will still be used for prediction\n");
        } else {
            printf("Using loaded global fallback model (trained on %d samples)\n", model.num_trained_samples);
        }
    } else {
        /* Try to load or train global model */
        if (flag_load && load_model(&model, model_path) == 0) {
            printf("Using loaded model (trained on %d samples)\n", model.num_trained_samples);
        } else {
            init_model(&model);
            train(&model, fights, num_fights);
            save_model(&model, model_path);
        }
    }

    if (flag_eval && flag_global) {
        evaluate_model(&model, fights, num_fights);
    } else if (flag_eval && !flag_global) {
        printf("Evaluation currently supported with --global mode. Run: ./ufc_nn --global --eval\n");
    }

    if (flag_matchup) {
        analyze_matchup(fights, num_fights);
    }
    
    /* Interactive prediction */
    if (flag_predict) {
        predict_interactive(&model, fights, num_fights, flag_predict_future);
    }
    
    free(fights);
    printf("\nGoodbye!\n");
    return 0;
}
