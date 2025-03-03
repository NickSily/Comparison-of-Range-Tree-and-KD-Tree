#include "../src/main.cpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/*
	Will Have to comment out main btw
	To check output (At the suffix_count directory):
		g++ -std=c++14 -Werror -Wuninitialized -o test test-unit/test.cpp && ./test
*/

TEST_CASE("Function: suffixCount 1", "[given]") {
	REQUIRE(2 == 2);
}

TEST_CASE("Function: suffixCount 2", "[given]") {
	REQUIRE(420 != 69);
}

// TEST_CASE("Function: suffixCount 4", "[o_hidden]") {
// 	REQUIRE(suffixCount("aaaaaa", 3) == ?);
// }

// TEST_CASE("Function: suffixCount 5", "[o_hidden]") {
// 	REQUIRE(suffixCount("aaabaaaa", 4) == ?);
// }
