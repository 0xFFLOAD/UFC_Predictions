#include "fighter_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"  /* assume cJSON is available */

/* simple djb2 string hash */
static unsigned long
hash_string(const char *str)
{
    unsigned long hash = 5381;
    int c;
    while ((c = *str++) != 0)
        hash = ((hash << 5) + hash) + (unsigned char)c;
    return hash;
}

/* global table definition */
Fighter *fighter_table[HASH_SIZE];

static Fighter *alloc_fighter(void)
{
    Fighter *f = calloc(1, sizeof(*f));
    return f;
}

static Fight *alloc_fight(void)
{
    Fight *f = calloc(1, sizeof(*f));
    return f;
}

/* helper to duplicate a string */
static char *dupstr(const char *s)
{
    size_t len = strlen(s) + 1;
    char *d = malloc(len);
    if (d)
        memcpy(d, s, len);
    return d;
}

int load_fighter_index(const char *json_path)
{
    FILE *fp = fopen(json_path, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    fread(buf, 1, size, fp);
    buf[size] = '\0';
    fclose(fp);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) {
        fprintf(stderr, "JSON parse error: %s\n", cJSON_GetErrorPtr());
        return -1;
    }

    /* determine feature names from first fighter encountered */
    cJSON *fighter_obj = NULL;
    cJSON_ArrayForEach(fighter_obj, root) {
        cJSON *data = fighter_obj->child;
        if (!data) continue;
        cJSON *mean_stats = cJSON_GetObjectItem(data, "mean_stats");
        if (mean_stats) {
            int nf = cJSON_GetArraySize(mean_stats);
            /* allocate arrays; we'll copy names later per fighter */
            /* store in a static array for simplicity */
            break;
        }
    }

    /* iterate all fighters and insert into hash table */
    cJSON *value;
    cJSON_ArrayForEach(fighter_obj, root) {
        const char *name = fighter_obj->string;
        if (!name) continue;
        Fighter *f = alloc_fighter();
        f->name = dupstr(name);

        cJSON *stats = cJSON_GetObjectItem(fighter_obj, "mean_stats");
        if (stats && cJSON_IsObject(stats)) {
            int nf = cJSON_GetArraySize(stats);
            f->n_features = nf;
            f->feature_names = malloc(sizeof(char *) * nf);
            f->mean_stats = malloc(sizeof(double) * nf);
            int idx = 0;
            cJSON *stat;
            cJSON_ArrayForEach(stat, stats) {
                f->feature_names[idx] = dupstr(stat->string);
                f->mean_stats[idx] = stat->valuedouble;
                idx++;
            }
        }

        cJSON *fights = cJSON_GetObjectItem(fighter_obj, "fights");
        if (fights && cJSON_IsArray(fights)) {
            cJSON *fight;
            Fight **lastp = &f->fights;
            cJSON_ArrayForEach(fight, fights) {
                cJSON *opp = cJSON_GetObjectItem(fight, "opponent");
                cJSON *won = cJSON_GetObjectItem(fight, "won");
                cJSON *wc = cJSON_GetObjectItem(fight, "weight_class");
                if (opp && cJSON_IsString(opp)) {
                    Fight *ff = alloc_fight();
                    ff->opponent = dupstr(opp->valuestring);
                    ff->won = (won && won->type == cJSON_True) ? 1 : 0;
                    ff->weight_class = wc && cJSON_IsString(wc) ? dupstr(wc->valuestring) : NULL;
                    *lastp = ff;
                    lastp = &ff->next;
                }
            }
        }

        /* insert into table */
        unsigned long h = hash_string(name) % HASH_SIZE;
        f->next = fighter_table[h];
        fighter_table[h] = f;
    }

    cJSON_Delete(root);
    return 0;
}

Fighter *lookup_fighter(const char *name)
{
    unsigned long h = hash_string(name) % HASH_SIZE;
    Fighter *f = fighter_table[h];
    while (f) {
        if (strcmp(f->name, name) == 0)
            return f;
        f = f->next;
    }
    return NULL;
}

void free_fighter_index(void)
{
    for (size_t i = 0; i < HASH_SIZE; i++) {
        Fighter *f = fighter_table[i];
        while (f) {
            Fighter *next = f->next;
            free(f->name);
            for (size_t j = 0; j < f->n_features; j++) {
                free(f->feature_names[j]);
            }
            free(f->feature_names);
            free(f->mean_stats);
            Fight *g = f->fights;
            while (g) {
                Fight *gn = g->next;
                free(g->opponent);
                free(g->weight_class);
                free(g);
                g = gn;
            }
            free(f);
            f = next;
        }
        fighter_table[i] = NULL;
    }
}

/* -------------------------------------------------------------------- */

/* simple activation helpers */
static float clip(float x) {
    if (x > 10.0f) return 10.0f;
    if (x < -10.0f) return -10.0f;
    return x;
}

static float tanh_approx(float x) {
    /* relying on math library tanhf; could implement faster but okay */
    return tanhf(x);
}

/* normalize raw features in-place according to the stored mean/std arrays */
float model_predict(const float *raw_features)
{
    float x[MODEL_INPUT_DIM];
    for (size_t i = 0; i < MODEL_INPUT_DIM; i++) {
        x[i] = (raw_features[i] - model_mean[i]) / model_std[i];
        x[i] = clip(x[i]);
    }
    /* first layer */
    float h1[MODEL_HIDDEN1];
    for (size_t i = 0; i < MODEL_HIDDEN1; i++) {
        float acc = model_b0[i];
        for (size_t j = 0; j < MODEL_INPUT_DIM; j++)
            acc += model_w0[i][j] * x[j];
        h1[i] = tanh_approx(acc);
    }
    /* second layer */
    float h2[MODEL_HIDDEN2];
    for (size_t i = 0; i < MODEL_HIDDEN2; i++) {
        float acc = model_b1[i];
        for (size_t j = 0; j < MODEL_HIDDEN1; j++)
            acc += model_w1[i][j] * h1[j];
        h2[i] = tanh_approx(acc);
    }
    /* output layer */
    float logit = model_b2[0];
    for (size_t j = 0; j < MODEL_HIDDEN2; j++)
        logit += model_w2[0][j] * h2[j];
    /* sigmoid */
    return 1.0f / (1.0f + expf(-logit));
}

int predict_fight(const char *f1, const char *f2, float *prob)
{
    Fighter *p1 = lookup_fighter(f1);
    Fighter *p2 = lookup_fighter(f2);
    if (!p1 || !p2)
        return -1;
    float input[MODEL_INPUT_DIM];
    for (size_t i = 0; i < MODEL_INPUT_DIM; i++) {
        float v1 = (i < p1->n_features) ? p1->mean_stats[i] : 0.0;
        float v2 = (i < p2->n_features) ? p2->mean_stats[i] : 0.0;
        input[i] = v1 - v2;
    }
    *prob = model_predict(input);
    return 0;
}

/* simple demo program when compiled as standalone */
#ifdef FIGHTER_INDEX_MAIN
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s fighter_index.json [fighter1 fighter2]\n", argv[0]);
        return 1;
    }
    if (load_fighter_index(argv[1]) != 0) {
        return 1;
    }
    if (argc >= 4) {
        const char *f1 = argv[2];
        const char *f2 = argv[3];
        float prob;
        if (predict_fight(f1, f2, &prob) == 0) {
            printf("%s vs %s -> probability red wins (first name): %.2f%%\n",
                   f1, f2, prob * 100.0f);
            printf("predicted winner: %s\n", prob > 0.5f ? f1 : f2);
        } else {
            printf("one or both fighters not found in index\n");
        }
    } else {
        const char *query = argc > 2 ? argv[2] : "Conor McGregor";
        Fighter *f = lookup_fighter(query);
        if (!f) {
            printf("%s not found\n", query);
        } else {
            printf("%s (%.0zu fights)\n", f->name, f->fights ? 1 : 0);
            for (Fight *g = f->fights; g; g = g->next) {
                printf("  vs %s %s (%s)\n", g->opponent,
                       g->won ? "won" : "lost",
                       g->weight_class ? g->weight_class : "");
            }
        }
    }
    free_fighter_index();
    return 0;
}
#endif
