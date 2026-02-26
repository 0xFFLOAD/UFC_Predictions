#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 4096
#define DATA_PATH_PRIMARY "data/TSLA.csv"
#define DATA_PATH_FALLBACK "../data/TSLA.csv"
#define MODEL_FILE "tsla_model.bin"

typedef struct {
    char date[16];
    double close;
} PriceRow;

typedef struct {
    double mean_return;
    double std_return;
    double last_close;
    char last_date[16];
    int samples;
} TeslaModel;

static double sigmoid(double x) {
    if (x > 40.0) return 1.0;
    if (x < -40.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

static double parse_number(const char *src) {
    char buffer[128];
    size_t out = 0;
    for (size_t i = 0; src[i] != '\0' && out + 1 < sizeof(buffer); i++) {
        if (src[i] != ',') {
            buffer[out++] = src[i];
        }
    }
    buffer[out] = '\0';
    return atof(buffer);
}

static int split_csv(const char *line, char fields[][128], int max_fields) {
    int field_idx = 0;
    int out_idx = 0;
    int in_quotes = 0;

    for (int i = 0; line[i] != '\0'; i++) {
        char ch = line[i];
        if (ch == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (ch == ',' && !in_quotes) {
            if (field_idx < max_fields) {
                fields[field_idx][out_idx] = '\0';
                field_idx++;
            }
            out_idx = 0;
            continue;
        }
        if (ch == '\n' || ch == '\r') {
            continue;
        }
        if (field_idx < max_fields && out_idx + 1 < 128) {
            fields[field_idx][out_idx++] = ch;
        }
    }

    if (field_idx < max_fields) {
        fields[field_idx][out_idx] = '\0';
        field_idx++;
    }

    return field_idx;
}

static int equals_ci(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static void trim_in_place(char *text) {
    if (!text || !text[0]) return;

    char *start = text;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }

    if ((unsigned char)start[0] == 0xEF &&
        (unsigned char)start[1] == 0xBB &&
        (unsigned char)start[2] == 0xBF) {
        start += 3;
    }

    char *end = start + strlen(start);
    while (end > start && isspace((unsigned char)*(end - 1))) {
        end--;
    }
    *end = '\0';

    if (start != text) {
        memmove(text, start, (size_t)(end - start) + 1);
    }
}

static int find_field_index(char fields[][128], int field_count, const char *name) {
    for (int i = 0; i < field_count; i++) {
        trim_in_place(fields[i]);
        if (equals_ci(fields[i], name)) {
            return i;
        }
    }
    return -1;
}

static const char *resolve_dataset_path(void) {
    static char resolved_path[128];
    const char *candidates[] = {DATA_PATH_PRIMARY, DATA_PATH_FALLBACK};
    const int candidate_count = (int)(sizeof(candidates) / sizeof(candidates[0]));

    for (int i = 0; i < candidate_count; i++) {
        FILE *probe = fopen(candidates[i], "r");
        if (probe) {
            fclose(probe);
            strncpy(resolved_path, candidates[i], sizeof(resolved_path) - 1);
            resolved_path[sizeof(resolved_path) - 1] = '\0';
            return resolved_path;
        }
    }

    strncpy(resolved_path, DATA_PATH_PRIMARY, sizeof(resolved_path) - 1);
    resolved_path[sizeof(resolved_path) - 1] = '\0';
    return resolved_path;
}

static const char *resolve_model_path(const char *argv0) {
    static char resolved_path[512];

    if (argv0 && argv0[0]) {
        const char *last_slash = strrchr(argv0, '/');
        if (last_slash) {
            size_t dir_len = (size_t)(last_slash - argv0);
            if (dir_len + 1 + strlen(MODEL_FILE) + 1 < sizeof(resolved_path)) {
                memcpy(resolved_path, argv0, dir_len);
                resolved_path[dir_len] = '/';
                strcpy(resolved_path + dir_len + 1, MODEL_FILE);
                return resolved_path;
            }
        }
    }

    strncpy(resolved_path, MODEL_FILE, sizeof(resolved_path) - 1);
    resolved_path[sizeof(resolved_path) - 1] = '\0';
    return resolved_path;
}

static int load_prices(const char *path, PriceRow *rows, int *count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return -1;
    }

    char line[2048];
    int n = 0;

    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    char header_fields[32][128] = {{0}};
    int header_count = split_csv(line, header_fields, 32);
    int date_idx = find_field_index(header_fields, header_count, "Date");
    int close_idx = find_field_index(header_fields, header_count, "Close");
    if (date_idx < 0 || close_idx < 0) {
        fprintf(stderr, "Error: CSV missing Date/Close columns in %s\n", path);
        fclose(f);
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        char fields[32][128] = {{0}};
        int field_count = split_csv(line, fields, 32);
        if (field_count <= close_idx || field_count <= date_idx) {
            continue;
        }
        if (n >= MAX_ROWS) {
            break;
        }

        strncpy(rows[n].date, fields[date_idx], sizeof(rows[n].date) - 1);
        rows[n].close = parse_number(fields[close_idx]);
        if (rows[n].close <= 0.0) {
            continue;
        }
        n++;
    }

    fclose(f);
    *count = n;
    return n > 1 ? 0 : -1;
}

static int train_model(TeslaModel *model, PriceRow *rows, int count) {
    if (count < 3) return -1;

    double sum = 0.0;
    double sq_sum = 0.0;
    int samples = 0;

    for (int i = 1; i < count; i++) {
        double prev = rows[i - 1].close;
        double curr = rows[i].close;
        if (prev <= 0.0) continue;
        double r = (curr - prev) / prev;
        sum += r;
        sq_sum += r * r;
        samples++;
    }

    if (samples < 2) return -1;

    model->mean_return = sum / samples;
    double variance = (sq_sum / samples) - (model->mean_return * model->mean_return);
    model->std_return = sqrt(variance > 1e-12 ? variance : 1e-12);
    model->last_close = rows[count - 1].close;
    strncpy(model->last_date, rows[count - 1].date, sizeof(model->last_date) - 1);
    model->samples = samples;
    return 0;
}

static int save_model(const TeslaModel *model, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(model, sizeof(TeslaModel), 1, f);
    fclose(f);
    return 0;
}

static int load_model(TeslaModel *model, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int ok = fread(model, sizeof(TeslaModel), 1, f) == 1 ? 0 : -1;
    fclose(f);
    return ok;
}

static void predict_and_print(const TeslaModel *model, const char *forecast_date) {
    double predicted_close = model->last_close * (1.0 + model->mean_return);
    double signal = model->std_return > 1e-8 ? model->mean_return / model->std_return : 0.0;
    double bullish_prob = sigmoid(signal) * 100.0;

    printf("\n=== Tesla Stock Forecast ===\n");
    printf("Forecast date      : %s\n", forecast_date && forecast_date[0] ? forecast_date : "NEXT");
    printf("Last known date    : %s\n", model->last_date);
    printf("Last close         : %.2f\n", model->last_close);
    printf("Expected return    : %.3f%%\n", model->mean_return * 100.0);
    printf("Predicted close    : %.2f\n", predicted_close);
    printf("Bullish probability: %.2f%%\n", bullish_prob);
}

int main(int argc, char **argv) {
    int flag_train = 0;
    int flag_load = 0;
    int flag_predict = 0;
    const char *forecast_date = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0) flag_train = 1;
        else if (strcmp(argv[i], "--load") == 0) flag_load = 1;
        else if (strcmp(argv[i], "--predict") == 0) flag_predict = 1;
        else if (strcmp(argv[i], "--date") == 0 && i + 1 < argc) {
            forecast_date = argv[++i];
        }
    }

    if (!flag_train && !flag_load) flag_train = 1;
    if (!flag_predict) flag_predict = 1;

    PriceRow rows[MAX_ROWS];
    int count = 0;
    const char *dataset_path = resolve_dataset_path();
    const char *model_path = resolve_model_path(argv[0]);
    if (load_prices(dataset_path, rows, &count) != 0) {
        fprintf(stderr, "Failed to load Tesla dataset\n");
        return 1;
    }

    TeslaModel model = {0};

    if (flag_load && load_model(&model, model_path) == 0) {
        printf("Loaded model from %s\n", model_path);
    } else {
        if (train_model(&model, rows, count) != 0) {
            fprintf(stderr, "Training failed\n");
            return 1;
        }
        if (save_model(&model, model_path) != 0) {
            fprintf(stderr, "Failed to save model\n");
            return 1;
        }
        printf("Trained model from %d samples and saved to %s\n", model.samples, model_path);
    }

    if (flag_predict) {
        predict_and_print(&model, forecast_date);
    }

    return 0;
}
