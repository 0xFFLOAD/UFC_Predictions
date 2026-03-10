#include <stdio.h>
#include <string.h>
#include <math.h>

#include "fighter_index_data.h"  /* static fighter list */
#include "model_weights.h"       /* network weights + feature names */

/* activation helpers copied from fighter_index.c earlier */
static float clip_val(float x) {
    if (x > 10.0f) return 10.0f;
    if (x < -10.0f) return -10.0f;
    return x;
}

static float tanh_approx(float x) {
    return tanhf(x);
}

float model_predict(const float *raw_features)
{
    float x[MODEL_INPUT_DIM];
    for (size_t i = 0; i < MODEL_INPUT_DIM; i++) {
        x[i] = (raw_features[i] - model_mean[i]) / model_std[i];
        x[i] = clip_val(x[i]);
    }
    float h1[MODEL_HIDDEN1];
    for (size_t i = 0; i < MODEL_HIDDEN1; i++) {
        float acc = model_b0[i];
        for (size_t j = 0; j < MODEL_INPUT_DIM; j++)
            acc += model_w0[i][j] * x[j];
        h1[i] = tanh_approx(acc);
    }
    float h2[MODEL_HIDDEN2];
    for (size_t i = 0; i < MODEL_HIDDEN2; i++) {
        float acc = model_b1[i];
        for (size_t j = 0; j < MODEL_HIDDEN1; j++)
            acc += model_w1[i][j] * h1[j];
        h2[i] = tanh_approx(acc);
    }
    float logit = model_b2[0];
    for (size_t j = 0; j < MODEL_HIDDEN2; j++)
        logit += model_w2[0][j] * h2[j];
    return 1.0f / (1.0f + expf(-logit));
}

int find_fighter(const char *name)
{
    for (size_t i = 0; i < n_fighters; i++) {
        if (strcmp(fighters[i].name, name) == 0)
            return (int)i;
    }
    return -1;
}

int predict_fight(const char *f1, const char *f2, float *prob)
{
    int i1 = find_fighter(f1);
    int i2 = find_fighter(f2);
    if (i1 < 0 || i2 < 0)
        return -1;
    Fighter *p1 = &fighters[i1];
    Fighter *p2 = &fighters[i2];
    float input[MODEL_INPUT_DIM];
    for (size_t i = 0; i < MODEL_INPUT_DIM; i++) {
        float v1 = (i < p1->n_features) ? p1->mean_stats[i] : 0.0;
        float v2 = (i < p2->n_features) ? p2->mean_stats[i] : 0.0;
        input[i] = v1 - v2;
    }
    *prob = model_predict(input);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s fighter1 fighter2\n", argv[0]);
        return 1;
    }
    const char *f1 = argv[1];
    const char *f2 = argv[2];
    float prob;
    if (predict_fight(f1, f2, &prob) == 0) {
        printf("%s vs %s -> probability %s wins: %.2f%%\n",
               f1, f2, f1, prob * 100.0f);
        printf("predicted winner: %s\n", prob > 0.5f ? f1 : f2);
    } else {
        printf("fighter(s) not found\n");
    }
    return 0;
}
