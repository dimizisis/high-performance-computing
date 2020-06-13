
#include <stdio.h>
#include <string.h>

char *best_shuffle(const char *s, int *diff);

int main(void){
	int i, d;
	const char *r, *t[] = {"abracadabra", "seesaw", "elk", "grrrrrr", "up", "a", 0};
	for (i = 0; t[i]; ++i) {
		r = best_shuffle(t[i], &d);
		printf("%s %s (%d)\n", t[i], r, d);
	}
	return 0;
}

char *best_shuffle(const char *s, int *diff){
	int i, j = 0, max = 0, l = strlen(s), cnt[128] = {0};
	char buf[256] = {0}, *r;
 
	for(i = 0; i < l; ++i)
		if (++cnt[(int)s[i]] > max) max = cnt[(int)s[i]];

	for(i = 0; i < 128; ++i)
		while (cnt[i]--) buf[j++] = i;
 
	r = strdup(s);
    for(i = 0; i < l; ++i)
        for(j = 0; j < l; ++j)
            if (r[i] == buf[j]) {
                r[i] = buf[(j + max) % l] & ~128;
                buf[j] |= 128;
                break;
            }

	*diff = 0;

	for(i = 0; i < l; ++i)
		*diff += r[i] == s[i];
 
	return r;
}