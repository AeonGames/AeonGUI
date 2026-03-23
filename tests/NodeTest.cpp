/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "aeongui/dom/Node.hpp"

namespace
{
    /// Minimal concrete Node for testing traversals.
    class TestNode : public AeonGUI::DOM::Node
    {
    public:
        explicit TestNode ( const std::string& aLabel, Node* aParent = nullptr )
            : Node ( aParent ), mLabel{aLabel} {}
        NodeType nodeType() const override
        {
            return ELEMENT_NODE;
        }
        const std::string& label() const
        {
            return mLabel;
        }
    private:
        std::string mLabel;
    };

    /// Build the following tree and return the root:
    ///
    ///        root
    ///       / | \.
    ///      A   B   C
    ///     / \     |
    ///    D   E    F
    ///
    std::unique_ptr<TestNode> MakeTree()
    {
        auto root = std::make_unique<TestNode> ( "root" );
        auto* a = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "A", root.get() ) ) );
        root->AddNode ( std::make_unique<TestNode> ( "B", root.get() ) );
        auto* c = static_cast<TestNode*> ( root->AddNode ( std::make_unique<TestNode> ( "C", root.get() ) ) );
        a->AddNode ( std::make_unique<TestNode> ( "D", a ) );
        a->AddNode ( std::make_unique<TestNode> ( "E", a ) );
        c->AddNode ( std::make_unique<TestNode> ( "F", c ) );
        return root;
    }

    std::vector<std::string> ExpectedPreOrder()
    {
        return {"root", "A", "D", "E", "B", "C", "F"};
    }
    std::vector<std::string> ExpectedPostOrder()
    {
        return {"D", "E", "A", "B", "F", "C", "root"};
    }

    void CollectLabel ( const AeonGUI::DOM::Node& aNode, std::vector<std::string>& aOut )
    {
        aOut.push_back ( static_cast<const TestNode&> ( aNode ).label() );
    }
    void CollectLabel ( AeonGUI::DOM::Node& aNode, std::vector<std::string>& aOut )
    {
        aOut.push_back ( static_cast<TestNode&> ( aNode ).label() );
    }
}

// ========================================================================
// TraverseDepthFirstPreOrder — single action
// ========================================================================

TEST ( NodeTest, TraverseDepthFirstPreOrder_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> visited;
    root->TraverseDepthFirstPreOrder ( [&visited] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPreOrder() );
}

TEST ( NodeTest, TraverseDepthFirstPreOrder_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> visited;
    constRoot->TraverseDepthFirstPreOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPreOrder() );
}

// ========================================================================
// TraverseDepthFirstPostOrder — single action
// ========================================================================

TEST ( NodeTest, TraverseDepthFirstPostOrder_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> visited;
    root->TraverseDepthFirstPostOrder ( [&visited] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPostOrder() );
}

TEST ( NodeTest, TraverseDepthFirstPostOrder_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> visited;
    constRoot->TraverseDepthFirstPostOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPostOrder() );
}

// ========================================================================
// TraverseDepthFirstPreOrder — preamble + postamble
// ========================================================================

TEST ( NodeTest, TraverseDepthFirstPreOrder_PreamblePostamble_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    root->TraverseDepthFirstPreOrder (
        [&pre] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    } );
    EXPECT_EQ ( pre, ExpectedPreOrder() );
    EXPECT_EQ ( post, ExpectedPostOrder() );
}

TEST ( NodeTest, TraverseDepthFirstPreOrder_PreamblePostamble_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    constRoot->TraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    } );
    EXPECT_EQ ( pre, ExpectedPreOrder() );
    EXPECT_EQ ( post, ExpectedPostOrder() );
}

// ========================================================================
// TraverseDepthFirstPreOrder — preamble + postamble + predicate
// ========================================================================

TEST ( NodeTest, TraverseDepthFirstPreOrder_Predicate_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    // Predicate skips node "A" — so D and E should not appear.
    root->TraverseDepthFirstPreOrder (
        [&pre] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( AeonGUI::DOM::Node & n ) -> bool
    {
        return static_cast<TestNode&> ( n ).label() != "A";
    } );
    std::vector<std::string> expectedPre {"root", "A", "B", "C", "F"};
    std::vector<std::string> expectedPost {"A", "B", "F", "C", "root"};
    EXPECT_EQ ( pre, expectedPre );
    EXPECT_EQ ( post, expectedPost );
}

TEST ( NodeTest, TraverseDepthFirstPreOrder_Predicate_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    constRoot->TraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( const AeonGUI::DOM::Node & n ) -> bool
    {
        return static_cast<const TestNode&> ( n ).label() != "A";
    } );
    std::vector<std::string> expectedPre {"root", "A", "B", "C", "F"};
    std::vector<std::string> expectedPost {"A", "B", "F", "C", "root"};
    EXPECT_EQ ( pre, expectedPre );
    EXPECT_EQ ( post, expectedPost );
}

// ========================================================================
// StackTraverseDepthFirstPreOrder — single action
// ========================================================================

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> visited;
    root->StackTraverseDepthFirstPreOrder ( [&visited] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPreOrder() );
}

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> visited;
    constRoot->StackTraverseDepthFirstPreOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPreOrder() );
}

// ========================================================================
// StackTraverseDepthFirstPostOrder — single action
// ========================================================================

TEST ( NodeTest, StackTraverseDepthFirstPostOrder_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> visited;
    root->StackTraverseDepthFirstPostOrder ( [&visited] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPostOrder() );
}

TEST ( NodeTest, StackTraverseDepthFirstPostOrder_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> visited;
    constRoot->StackTraverseDepthFirstPostOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ExpectedPostOrder() );
}

// ========================================================================
// StackTraverseDepthFirstPreOrder — preamble + postamble
// ========================================================================

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_PreamblePostamble_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    root->StackTraverseDepthFirstPreOrder (
        [&pre] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    } );
    EXPECT_EQ ( pre, ExpectedPreOrder() );
    EXPECT_EQ ( post, ExpectedPostOrder() );
}

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_PreamblePostamble_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    constRoot->StackTraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    } );
    EXPECT_EQ ( pre, ExpectedPreOrder() );
    EXPECT_EQ ( post, ExpectedPostOrder() );
}

// ========================================================================
// StackTraverseDepthFirstPreOrder — preamble + postamble + predicate
// ========================================================================

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_Predicate_NonConst )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    root->StackTraverseDepthFirstPreOrder (
        [&pre] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( AeonGUI::DOM::Node & n ) -> bool
    {
        return static_cast<TestNode&> ( n ).label() != "A";
    } );
    std::vector<std::string> expectedPre {"root", "A", "B", "C", "F"};
    std::vector<std::string> expectedPost {"A", "B", "F", "C", "root"};
    EXPECT_EQ ( pre, expectedPre );
    EXPECT_EQ ( post, expectedPost );
}

TEST ( NodeTest, StackTraverseDepthFirstPreOrder_Predicate_Const )
{
    auto root = MakeTree();
    const AeonGUI::DOM::Node* constRoot = root.get();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    constRoot->StackTraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( const AeonGUI::DOM::Node & n ) -> bool
    {
        return static_cast<const TestNode&> ( n ).label() != "A";
    } );
    std::vector<std::string> expectedPre {"root", "A", "B", "C", "F"};
    std::vector<std::string> expectedPost {"A", "B", "F", "C", "root"};
    EXPECT_EQ ( pre, expectedPre );
    EXPECT_EQ ( post, expectedPost );
}

// ========================================================================
// Verify iterator-based and stack-based produce identical results
// ========================================================================

TEST ( NodeTest, IteratorAndStackPreOrderMatch )
{
    auto root = MakeTree();
    std::vector<std::string> iterVisited;
    std::vector<std::string> stackVisited;
    root->TraverseDepthFirstPreOrder ( [&iterVisited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterVisited );
    } );
    root->StackTraverseDepthFirstPreOrder ( [&stackVisited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackVisited );
    } );
    EXPECT_EQ ( iterVisited, stackVisited );
}

TEST ( NodeTest, IteratorAndStackPostOrderMatch )
{
    auto root = MakeTree();
    std::vector<std::string> iterVisited;
    std::vector<std::string> stackVisited;
    root->TraverseDepthFirstPostOrder ( [&iterVisited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterVisited );
    } );
    root->StackTraverseDepthFirstPostOrder ( [&stackVisited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackVisited );
    } );
    EXPECT_EQ ( iterVisited, stackVisited );
}

TEST ( NodeTest, IteratorAndStackPreamblePostambleMatch )
{
    auto root = MakeTree();
    std::vector<std::string> iterPre, iterPost, stackPre, stackPost;
    root->TraverseDepthFirstPreOrder (
        [&iterPre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterPre );
    },
    [&iterPost] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterPost );
    } );
    root->StackTraverseDepthFirstPreOrder (
        [&stackPre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackPre );
    },
    [&stackPost] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackPost );
    } );
    EXPECT_EQ ( iterPre, stackPre );
    EXPECT_EQ ( iterPost, stackPost );
}

TEST ( NodeTest, IteratorAndStackPredicateMatch )
{
    auto root = MakeTree();
    auto predicate = [] ( const AeonGUI::DOM::Node & n ) -> bool
    {
        return static_cast<const TestNode&> ( n ).label() != "C";
    };
    std::vector<std::string> iterPre, iterPost, stackPre, stackPost;
    root->TraverseDepthFirstPreOrder (
        [&iterPre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterPre );
    },
    [&iterPost] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, iterPost );
    },
    predicate );
    root->StackTraverseDepthFirstPreOrder (
        [&stackPre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackPre );
    },
    [&stackPost] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, stackPost );
    },
    predicate );
    EXPECT_EQ ( iterPre, stackPre );
    EXPECT_EQ ( iterPost, stackPost );
}

// ========================================================================
// Edge cases
// ========================================================================

TEST ( NodeTest, SingleNodePreOrder )
{
    TestNode root ( "only" );
    std::vector<std::string> visited;
    root.TraverseDepthFirstPreOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ( std::vector<std::string> {"only"} ) );
}

TEST ( NodeTest, SingleNodePostOrder )
{
    TestNode root ( "only" );
    std::vector<std::string> visited;
    root.TraverseDepthFirstPostOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ( std::vector<std::string> {"only"} ) );
}

TEST ( NodeTest, SingleNodeStackPreOrder )
{
    TestNode root ( "only" );
    std::vector<std::string> visited;
    root.StackTraverseDepthFirstPreOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ( std::vector<std::string> {"only"} ) );
}

TEST ( NodeTest, SingleNodeStackPostOrder )
{
    TestNode root ( "only" );
    std::vector<std::string> visited;
    root.StackTraverseDepthFirstPostOrder ( [&visited] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, visited );
    } );
    EXPECT_EQ ( visited, ( std::vector<std::string> {"only"} ) );
}

TEST ( NodeTest, PredicateRejectingRoot )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    root->TraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( const AeonGUI::DOM::Node& ) -> bool
    {
        return false;
    } );
    EXPECT_TRUE ( pre.empty() );
    EXPECT_TRUE ( post.empty() );
}

TEST ( NodeTest, StackPredicateRejectingRoot )
{
    auto root = MakeTree();
    std::vector<std::string> pre;
    std::vector<std::string> post;
    root->StackTraverseDepthFirstPreOrder (
        [&pre] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, pre );
    },
    [&post] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, post );
    },
    [] ( const AeonGUI::DOM::Node& ) -> bool
    {
        return false;
    } );
    EXPECT_TRUE ( pre.empty() );
    EXPECT_TRUE ( post.empty() );
}

// ========================================================================
// Reentrant safety — stack-based can be called inside iterator-based
// ========================================================================

TEST ( NodeTest, StackTraversalIsReentrantSafe )
{
    auto root = MakeTree();
    std::vector<std::string> innerResults;
    // During an iterator-based traversal, do a stack-based traversal
    // from the same root. The stack-based one should still work correctly
    // without corrupting the outer traversal.
    std::vector<std::string> outerResults;
    root->TraverseDepthFirstPreOrder (
        [&root, &outerResults, &innerResults] ( const AeonGUI::DOM::Node & n )
    {
        CollectLabel ( n, outerResults );
        if ( static_cast<const TestNode&> ( n ).label() == "B" )
        {
            innerResults.clear();
            root->StackTraverseDepthFirstPreOrder ( [&innerResults] ( const AeonGUI::DOM::Node & inner )
            {
                CollectLabel ( inner, innerResults );
            } );
        }
    } );
    // Inner traversal should have visited the full tree.
    EXPECT_EQ ( innerResults, ExpectedPreOrder() );
    // Outer traversal should also have visited the full tree.
    EXPECT_EQ ( outerResults, ExpectedPreOrder() );
}
