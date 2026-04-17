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
#include "aeongui/dom/Text.hpp"

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

// ========================================================================
// textContent — getter
// ========================================================================

TEST ( NodeTest, TextContent_ElementWithNoChildren )
{
    TestNode root ( "empty" );
    EXPECT_EQ ( root.textContent(), "" );
}

TEST ( NodeTest, TextContent_ElementWithSingleTextChild )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "hello", &root ) );
    EXPECT_EQ ( root.textContent(), "hello" );
}

TEST ( NodeTest, TextContent_ElementWithMultipleTextChildren )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "hello ", &root ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "world", &root ) );
    // textContent concatenates the data of each child Text node.
    EXPECT_EQ ( root.textContent(), "hello world" );
}

TEST ( NodeTest, TextContent_NestedElements )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "A", &root ) );
    auto* child = static_cast<TestNode*> ( root.AddNode ( std::make_unique<TestNode> ( "child", &root ) ) );
    child->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "B", child ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "C", &root ) );
    // textContent recurses into nested elements.
    EXPECT_EQ ( root.textContent(), "ABC" );
}

TEST ( NodeTest, TextContent_TextNodeReturnsSelf )
{
    AeonGUI::DOM::Text textNode ( "direct", nullptr );
    EXPECT_EQ ( textNode.textContent(), "direct" );
}

// ========================================================================
// setTextContent — setter
// ========================================================================

TEST ( NodeTest, SetTextContent_OnEmptyElement )
{
    TestNode root ( "parent" );
    root.setTextContent ( "new text" );
    EXPECT_EQ ( root.textContent(), "new text" );
    EXPECT_EQ ( root.childNodes().size(), 1u );
    EXPECT_EQ ( root.childNodes().front()->nodeType(), AeonGUI::DOM::Node::TEXT_NODE );
}

TEST ( NodeTest, SetTextContent_UpdatesExistingTextNode )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "old", &root ) );
    const AeonGUI::DOM::Node* originalChild = root.childNodes().front().get();
    root.setTextContent ( "new" );
    EXPECT_EQ ( root.textContent(), "new" );
    // Must reuse the same Text node, not create a new one.
    EXPECT_EQ ( root.childNodes().front().get(), originalChild );
    EXPECT_EQ ( root.childNodes().size(), 1u );
}

TEST ( NodeTest, SetTextContent_ReusesFirstTextNodeAmongMixed )
{
    TestNode root ( "parent" );
    auto* child = static_cast<TestNode*> ( root.AddNode ( std::make_unique<TestNode> ( "elem", &root ) ) );
    child->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "nested", child ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "direct1", &root ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "direct2", &root ) );
    EXPECT_EQ ( root.childNodes().size(), 3u );
    root.setTextContent ( "updated" );
    // Extra Text removed, first Text reused, element child with its own text untouched.
    EXPECT_EQ ( root.childNodes().size(), 2u );
    EXPECT_EQ ( root.textContent(), "nestedupdated" );
}

TEST ( NodeTest, SetTextContent_EmptyStringOnExistingText )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "something", &root ) );
    root.setTextContent ( "" );
    // Empty string should set the Text node to empty, not remove it.
    EXPECT_EQ ( root.textContent(), "" );
}

TEST ( NodeTest, SetTextContent_MultipleSetsReuseNode )
{
    TestNode root ( "parent" );
    root.setTextContent ( "first" );
    const AeonGUI::DOM::Node* child = root.childNodes().front().get();
    root.setTextContent ( "second" );
    EXPECT_EQ ( root.textContent(), "second" );
    EXPECT_EQ ( root.childNodes().front().get(), child );
    root.setTextContent ( "third" );
    EXPECT_EQ ( root.textContent(), "third" );
    EXPECT_EQ ( root.childNodes().front().get(), child );
}

TEST ( NodeTest, SetTextContent_RemovesExtraTextNodes )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "first", &root ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "second", &root ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "third", &root ) );
    EXPECT_EQ ( root.childNodes().size(), 3u );
    const AeonGUI::DOM::Node* kept = root.childNodes().front().get();
    root.setTextContent ( "only" );
    // Extra Text nodes should be removed, first one reused.
    EXPECT_EQ ( root.childNodes().size(), 1u );
    EXPECT_EQ ( root.childNodes().front().get(), kept );
    EXPECT_EQ ( root.textContent(), "only" );
}

TEST ( NodeTest, SetTextContent_RemovesExtraTextNodesAmongElements )
{
    TestNode root ( "parent" );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "A", &root ) );
    root.AddNode ( std::make_unique<TestNode> ( "elem", &root ) );
    root.AddNode ( std::make_unique<AeonGUI::DOM::Text> ( "B", &root ) );
    EXPECT_EQ ( root.childNodes().size(), 3u );
    root.setTextContent ( "updated" );
    // Element child preserved, extra Text removed, first Text reused.
    EXPECT_EQ ( root.childNodes().size(), 2u );
    EXPECT_EQ ( root.textContent(), "updated" );
}

// ========================================================================
// Text::data / setData
// ========================================================================

TEST ( NodeTest, TextData_GetAndSet )
{
    AeonGUI::DOM::Text textNode ( "initial", nullptr );
    EXPECT_EQ ( textNode.data(), "initial" );
    textNode.setData ( "changed" );
    EXPECT_EQ ( textNode.data(), "changed" );
    EXPECT_EQ ( textNode.textContent(), "changed" );
}
