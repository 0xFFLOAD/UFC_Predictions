#ifndef FIGHTER_INDEX_H
#define FIGHTER_INDEX_H

#include <stddef.h>

/* one entry in the linked list of bouts */
typedef struct Fight {
    char *opponent;            /* heap-allocated string */
    int won;                   /* 1 if this fighter won the bout */
    char *weight_class;        /* heap-allocated or interned string */
    struct Fight *next;
} Fight;

/* per-fighter record */
typedef struct Fighter {
    char *name;                /* heap-allocated fighter name */
    double *mean_stats;        /* array of length n_features */
    char **feature_names;      /* parallel array, owned by caller */
    size_t n_features;
    Fight *fights;             /* singly-linked list of previous opponents */
    struct Fighter *next;      /* for chaining inside a hash bucket */
} Fighter;

/* very simple hash table keyed by fighter name */
#define HASH_SIZE 4096
extern Fighter *fighter_table[HASH_SIZE];

/* load the JSON created by export_fighter_index.py; returns 0 on success */
int load_fighter_index(const char *json_path);

/* look up a fighter by name (case-sensitive); returns NULL if not found */
Fighter *lookup_fighter(const char *name);

/* free all memory allocated by load_fighter_index */
void free_fighter_index(void);

/* --- model inference -------------------------------------------------- */

/* input dimension and feature names come from the generated weights header
   (MODEL_INPUT_DIM and model_feature_names).  The caller should compute an
   input vector by subtracting fighter1 minus fighter2 statistics; see
   predict_fight below. */

/* normalize and run the network; returns value in [0,1] giving probability
   that the "red" side wins. */
float model_predict(const float *raw_features);

/* convenience: look up two fighters and fill ``*prob`` with the predicted
   probability that `f1` beats `f2`.  Returns 0 on success, -1 if either
   fighter is missing. */
int predict_fight(const char *f1, const char *f2, float *prob);

#endif /* FIGHTER_INDEX_H */
